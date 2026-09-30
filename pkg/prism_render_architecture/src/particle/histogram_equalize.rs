//! Generic `histogram` equalization and contrast-limited adaptive `histogram`
//! equalization (`CLAHE`) — the device-free `CPU` reference for the particle
//! shading path's contrast-normalization contract (design §16-§21).
//!
//! Histogram equalization redistributes a scalar channel (a mask, an emissive
//! ramp, a `luminance`-independent contrast field) so that its cumulative
//! distribution becomes as close to linear as the discrete bins allow, which
//! stretches the busy part of the value range and compresses the empty tails.
//! The pipeline is the textbook one: bin the samples into a `histogram`, prefix
//! sum the bins into a cumulative distribution function (`CDF`), normalize that
//! `CDF` into a lookup table (`LUT`) that maps each input bin to an output
//! level, and finally query the `LUT` per sample. `CLAHE` adds one extra step —
//! [`clip_histogram`] caps every over-tall bin at a clip limit and hands the
//! removed mass back out uniformly — so a single dominant value can no longer
//! blow the contrast curve out of shape.
//!
//! This module owns the `CPU`-verifiable maths of that contract so a future
//! `GPU` compute kernel (which fills a `std430` `u32` `histogram` buffer, prefix
//! scans it, and samples the `LUT` in a fragment or compute pass) has a
//! bit-checkable reference to match. [`gpu_storage_bytes`] reports the `std430`
//! byte size that `histogram` buffer must reserve, reusing the shared stride
//! primitives from [`crate::particle::gpu_layout`].
//!
//! # No transcendental functions
//!
//! The determinism-locked contract layer forbids `sin`/`cos`/`exp`/`ln`/`log2`
//! and friends. Nothing here needs them: binning, prefix sums, clipping, and
//! `LUT` construction are integer arithmetic plus scalar add / subtract /
//! multiply / guarded divide and a single `f32::floor` for bin placement and
//! level quantization. No lookup depends on the magnitude of the input, so the
//! `CPU` and `GPU` paths stay reproducible bit for bit.
//!
//! # Orthogonality
//!
//! [`super::luminance_hist`] owns the *exposure* front end: it places a
//! `luminance` on a `log2`-`EV` axis and reads an auto-exposure multiplier back
//! out. This module deliberately does neither. It performs no `log2`, computes
//! no `EV`, and knows nothing about exposure; it is the general-purpose `CDF`
//! equalization and `CLAHE` contract only. The two modules share no private
//! helper and never import each other.

use crate::particle::gpu_layout::{storage_bytes, U32_STRIDE};
use alloc::vec::Vec;

/// Absolute tolerance used to treat an f32 span or denominator as degenerate.
///
/// Value ranges narrower than this collapse to a single output, and relational
/// comparisons against this constant replace bare `==`/`!=` on floating-point
/// values throughout the module.
pub const CMP_EPS: f32 = 1e-6;

/// Default `histogram` bin count: an 8-bit-style 256-level distribution.
pub const DEFAULT_BINS: u32 = 256;

/// Default `CLAHE` clip factor, expressed as a multiple of the mean bin height.
///
/// A factor of `4.0` allows any bin to stand four times taller than the flat
/// average before its surplus is clipped and redistributed, a common default
/// for adaptive equalization.
pub const DEFAULT_CLIP_FACTOR: f32 = 4.0;

/// Reinterprets a [`u32`] count as an [`f32`] for the normalization arithmetic.
#[expect(
    clippy::cast_precision_loss,
    reason = "contract histogram counts stay far below f32's 2^24 exact-integer range"
)]
fn u32_as_f32(value: u32) -> f32 {
    value as f32
}

/// Reinterprets a [`usize`] bin count as an [`f32`] for bin-index arithmetic.
#[expect(
    clippy::cast_precision_loss,
    reason = "contract bin counts stay far below f32's 2^24 exact-integer range"
)]
fn usize_as_f32(value: usize) -> f32 {
    value as f32
}

/// Converts a pre-floored, clamped, non-negative [`f32`] into a bin index.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the argument is floored and clamped into [0, len) before the cast"
)]
fn f32_as_usize(value: f32) -> usize {
    value as usize
}

/// Converts a floored, non-negative [`f32`] clip limit into a [`u32`] count.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the argument is floored and non-negative; the as-cast saturates on overflow"
)]
fn f32_as_u32(value: f32) -> u32 {
    value as u32
}

/// Places a single `sample` into one of `bins` uniform buckets spanning
/// `[range_min, range_max]`.
///
/// Callers guarantee `bins > 0`. Samples at or below `range_min` land in bin
/// `0`, samples at or above `range_max` land in the last bin, and a degenerate
/// (zero-width) span sends everything to bin `0`. A `NaN` sample floors to bin
/// `0` rather than panicking.
fn bin_index(sample: f32, bins: usize, range_min: f32, range_max: f32) -> usize {
    let last = bins - 1;
    let span = range_max - range_min;
    if span <= CMP_EPS {
        return 0;
    }
    if sample <= range_min {
        return 0;
    }
    if sample >= range_max {
        return last;
    }
    let normalized = (sample - range_min) / span;
    let scaled = normalized * usize_as_f32(bins);
    let idx = f32_as_usize(scaled.floor());
    idx.min(last)
}

/// Configuration for a `histogram`-equalization or `CLAHE` pass (design §16).
///
/// [`bins`](Self::bins) sets the resolution of the distribution and
/// [`clip_limit`](Self::clip_limit) is the `CLAHE` clip *factor*: a multiple of
/// the mean bin height, resolved to an absolute per-bin cap by
/// [`absolute_clip_limit`](Self::absolute_clip_limit). A factor of `1.0` clips
/// to the flat average (maximum contrast limiting); larger factors permit more
/// local contrast before clipping.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HistConfig {
    /// Number of `histogram` bins spanning the working value range.
    pub bins: u32,
    /// `CLAHE` clip factor as a multiple of the mean bin height.
    pub clip_limit: f32,
}

impl HistConfig {
    /// Builds a configuration from an explicit bin count and clip factor.
    #[must_use]
    pub const fn new(bins: u32, clip_limit: f32) -> Self {
        Self { bins, clip_limit }
    }

    /// Resolves the clip *factor* into an absolute per-bin cap for
    /// [`clip_histogram`], given the total number of binned samples.
    ///
    /// The cap is `max(1, floor(clip_limit * total_samples / bins))`, i.e. the
    /// clip factor times the mean bin height, guarded to at least one so a
    /// valid `histogram` is never fully flattened to zero. A zero bin count
    /// yields `0`.
    #[must_use]
    pub fn absolute_clip_limit(&self, total_samples: u32) -> u32 {
        if self.bins == 0 {
            return 0;
        }
        let mean = total_samples / self.bins;
        let factor = self.clip_limit.max(1.0);
        let scaled = (factor * u32_as_f32(mean)).floor();
        f32_as_u32(scaled).max(1)
    }
}

impl Default for HistConfig {
    fn default() -> Self {
        Self::new(DEFAULT_BINS, DEFAULT_CLIP_FACTOR)
    }
}

/// Bins `samples` into `bins` uniform buckets spanning `[range_min, range_max]`.
///
/// Out-of-range samples are clamped into the two end buckets, a zero-width span
/// counts everything into bin `0`, and a `bins` of `0` returns an empty
/// `histogram`. Counts saturate rather than overflowing.
#[must_use]
pub fn build_histogram(samples: &[f32], bins: usize, range_min: f32, range_max: f32) -> Vec<u32> {
    let mut hist = Vec::new();
    if bins == 0 {
        return hist;
    }
    hist.resize(bins, 0u32);
    for &sample in samples {
        let idx = bin_index(sample, bins, range_min, range_max);
        hist[idx] = hist[idx].saturating_add(1);
    }
    hist
}

/// Prefix-sums a `histogram` into its cumulative distribution function (`CDF`).
///
/// Entry `i` of the result is the running total of `hist[0..=i]`, so the final
/// entry equals the summed sample count. The running total saturates rather
/// than overflowing.
#[must_use]
pub fn cumulative_distribution(hist: &[u32]) -> Vec<u32> {
    let mut cdf = Vec::with_capacity(hist.len());
    let mut acc: u32 = 0;
    for &count in hist {
        acc = acc.saturating_add(count);
        cdf.push(acc);
    }
    cdf
}

/// Normalizes a `CDF` into an equalization `LUT` whose entries lie in `[0, 1]`.
///
/// `total` is the sample count the `CDF` saturates to (its last entry) and
/// `out_levels` is the number of discrete output levels to quantize onto. The
/// classic map subtracts the smallest non-zero `CDF` value (`cdf_min`) before
/// normalizing so the darkest occupied bin maps to `0`:
/// `lut[i] = (cdf[i] - cdf_min) / (total - cdf_min)`.
///
/// When `out_levels > 1` the normalized value is snapped to the nearest of
/// `out_levels` evenly spaced levels; otherwise it is returned continuously. A
/// degenerate denominator (a flat or empty distribution) yields an all-zero
/// `LUT`, and an empty `CDF` yields an empty `LUT`.
#[must_use]
pub fn equalization_lut(cdf: &[u32], total: u32, out_levels: u32) -> Vec<f32> {
    let mut lut = Vec::with_capacity(cdf.len());
    if cdf.is_empty() {
        return lut;
    }
    let mut cdf_min: u32 = 0;
    for &value in cdf {
        if value > 0 {
            cdf_min = value;
            break;
        }
    }
    let denom = total.saturating_sub(cdf_min);
    if denom == 0 {
        lut.resize(cdf.len(), 0.0);
        return lut;
    }
    let denom_f = u32_as_f32(denom);
    let quantize = out_levels > 1;
    let max_level_f = if quantize {
        u32_as_f32(out_levels - 1)
    } else {
        1.0
    };
    for &value in cdf {
        let numer = value.saturating_sub(cdf_min);
        let raw = (u32_as_f32(numer) / denom_f).clamp(0.0, 1.0);
        let mapped = if quantize {
            let level = (raw * max_level_f + 0.5).floor().clamp(0.0, max_level_f);
            level / max_level_f
        } else {
            raw
        };
        lut.push(mapped);
    }
    lut
}

/// Maps one `sample` through an equalization `LUT` back into
/// `[range_min, range_max]`.
///
/// The sample is binned into `lut.len()` buckets spanning the range, its `LUT`
/// entry (a normalized level in `[0, 1]`) is read, and that level is expanded
/// back across the range: `range_min + lut[bin] * (range_max - range_min)`. An
/// empty `LUT` returns the sample unchanged, and a zero-width range returns
/// `range_min`.
#[must_use]
pub fn apply_equalization(sample: f32, lut: &[f32], range_min: f32, range_max: f32) -> f32 {
    if lut.is_empty() {
        return sample;
    }
    let span = range_max - range_min;
    if span <= CMP_EPS {
        return range_min;
    }
    let idx = bin_index(sample, lut.len(), range_min, range_max);
    let normalized = lut[idx];
    range_min + normalized * span
}

/// Applies the `CLAHE` clip-and-redistribute step to a `histogram` in place.
///
/// Every bin taller than `clip_limit` is capped at the limit, and the total
/// clipped surplus is handed back out uniformly: each bin gains
/// `surplus / bins`, and the leftover `surplus % bins` is spread one count at a
/// time across the leading bins so no mass is lost (up to saturation). An empty
/// `histogram` is left untouched. Counts saturate rather than overflowing.
pub fn clip_histogram(hist: &mut [u32], clip_limit: u32) {
    let bins = hist.len();
    if bins == 0 {
        return;
    }
    let mut surplus: u32 = 0;
    for count in hist.iter_mut() {
        if *count > clip_limit {
            surplus = surplus.saturating_add(*count - clip_limit);
            *count = clip_limit;
        }
    }
    let bins_u32 = u32::try_from(bins).unwrap_or(u32::MAX);
    let per_bin = surplus / bins_u32;
    let remainder = surplus % bins_u32;
    for (i, count) in hist.iter_mut().enumerate() {
        *count = count.saturating_add(per_bin);
        let idx = u32::try_from(i).unwrap_or(u32::MAX);
        if idx < remainder {
            *count = count.saturating_add(1);
        }
    }
}

/// Reports the `std430` `GPU` storage byte size for a `u32` `histogram` buffer
/// of `bin_count` bins.
///
/// Reuses [`crate::particle::gpu_layout::storage_bytes`] so an empty
/// `histogram` still reserves one element (a `WebGPU` storage binding may not
/// be zero-sized) and the multiplication saturates instead of overflowing.
#[must_use]
pub fn gpu_storage_bytes(bin_count: usize) -> usize {
    storage_bytes(U32_STRIDE, bin_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_EPS: f32 = 1e-4;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= TEST_EPS
    }

    fn is_nondecreasing(values: &[f32]) -> bool {
        values.windows(2).all(|w| w[1] >= w[0] - TEST_EPS)
    }

    fn total_count(hist: &[u32]) -> u32 {
        hist.iter().copied().fold(0u32, u32::saturating_add)
    }

    #[test]
    fn hist_config_new_stores_fields() {
        let cfg = HistConfig::new(64, 2.5);
        assert_eq!(cfg.bins, 64);
        assert!(close(cfg.clip_limit, 2.5));
    }

    #[test]
    fn hist_config_default_matches_named_constants() {
        let cfg = HistConfig::default();
        assert_eq!(cfg.bins, DEFAULT_BINS);
        assert!(close(cfg.clip_limit, DEFAULT_CLIP_FACTOR));
    }

    #[test]
    fn absolute_clip_limit_scales_mean_by_factor() {
        let cfg = HistConfig::new(4, 2.0);
        // mean height = 100 / 4 = 25; factor 2.0 -> 50.
        assert_eq!(cfg.absolute_clip_limit(100), 50);
    }

    #[test]
    fn absolute_clip_limit_is_at_least_one() {
        let cfg = HistConfig::new(8, 1.0);
        assert_eq!(cfg.absolute_clip_limit(0), 1);
    }

    #[test]
    fn absolute_clip_limit_zero_bins_is_zero() {
        let cfg = HistConfig::new(0, 4.0);
        assert_eq!(cfg.absolute_clip_limit(1000), 0);
    }

    #[test]
    fn absolute_clip_limit_floors_factor_to_one() {
        let cfg = HistConfig::new(4, 0.1);
        // factor clamped up to 1.0; mean = 40 / 4 = 10.
        assert_eq!(cfg.absolute_clip_limit(40), 10);
    }

    #[test]
    fn build_histogram_counts_each_bucket() {
        let samples = [0.1, 0.2, 0.6, 0.9, 0.95];
        let bins = 4usize;
        let hist = build_histogram(&samples, bins, 0.0, 1.0);
        assert_eq!(hist.len(), bins);
        assert_eq!(total_count(&hist), 5);
        // 0.1,0.2 -> bin0; 0.6 -> bin2; 0.9,0.95 -> bin3.
        assert_eq!(hist[0], 2);
        assert_eq!(hist[1], 0);
        assert_eq!(hist[2], 1);
        assert_eq!(hist[3], 2);
    }

    #[test]
    fn build_histogram_zero_bins_is_empty() {
        let hist = build_histogram(&[0.5, 0.7], 0, 0.0, 1.0);
        assert_eq!(hist.len(), 0);
    }

    #[test]
    fn build_histogram_clamps_below_range_to_first_bin() {
        let hist = build_histogram(&[-5.0, -0.001], 3, 0.0, 1.0);
        assert_eq!(hist[0], 2);
        assert_eq!(total_count(&hist), 2);
    }

    #[test]
    fn build_histogram_clamps_above_range_to_last_bin() {
        let bins = 3usize;
        let hist = build_histogram(&[2.0, 9.0, 100.0], bins, 0.0, 1.0);
        assert_eq!(hist[bins - 1], 3);
        assert_eq!(total_count(&hist), 3);
    }

    #[test]
    fn build_histogram_degenerate_range_uses_first_bin() {
        let hist = build_histogram(&[0.3, 0.3, 0.3], 5, 0.3, 0.3);
        assert_eq!(hist[0], 3);
        assert_eq!(total_count(&hist), 3);
    }

    #[test]
    fn build_histogram_preserves_total_sample_count() {
        let samples = [0.0, 0.25, 0.5, 0.75, 1.0, 0.33, 0.66];
        let hist = build_histogram(&samples, 8, 0.0, 1.0);
        assert_eq!(total_count(&hist), u32::try_from(samples.len()).unwrap());
    }

    #[test]
    fn build_histogram_nan_sample_does_not_panic() {
        let hist = build_histogram(&[f32::NAN, 0.5], 4, 0.0, 1.0);
        assert_eq!(total_count(&hist), 2);
    }

    #[test]
    fn cumulative_distribution_is_running_sum() {
        let hist = [1u32, 2, 3, 4];
        let cdf = cumulative_distribution(&hist);
        assert_eq!(cdf, [1u32, 3, 6, 10]);
    }

    #[test]
    fn cumulative_distribution_last_equals_total() {
        let hist = [5u32, 0, 7, 2, 1];
        let cdf = cumulative_distribution(&hist);
        let last = cdf.len() - 1;
        assert_eq!(cdf[last], total_count(&hist));
    }

    #[test]
    fn cumulative_distribution_empty_is_empty() {
        let cdf = cumulative_distribution(&[]);
        assert_eq!(cdf.len(), 0);
    }

    #[test]
    fn equalization_lut_is_monotonic_nondecreasing() {
        let hist = build_histogram(&[0.1, 0.2, 0.5, 0.5, 0.9], 8, 0.0, 1.0);
        let total = total_count(&hist);
        let cdf = cumulative_distribution(&hist);
        let lut = equalization_lut(&cdf, total, 256);
        assert_eq!(lut.len(), cdf.len());
        assert!(is_nondecreasing(&lut));
    }

    #[test]
    fn equalization_lut_stays_in_unit_interval() {
        let cdf = [0u32, 2, 5, 9, 12];
        let lut = equalization_lut(&cdf, 12, 64);
        let all_in_range = lut.iter().all(|&v| (0.0..=1.0).contains(&v));
        assert!(all_in_range);
        let last = lut.len() - 1;
        assert!(close(lut[last], 1.0));
    }

    #[test]
    fn equalization_lut_empty_cdf_is_empty() {
        let lut = equalization_lut(&[], 0, 256);
        assert_eq!(lut.len(), 0);
    }

    #[test]
    fn equalization_lut_flat_distribution_is_all_zero() {
        let cdf = [4u32, 4, 4];
        // total equals cdf_min, so the denominator degenerates.
        let lut = equalization_lut(&cdf, 4, 256);
        let all_zero = lut.iter().all(|&v| close(v, 0.0));
        assert!(all_zero);
    }

    #[test]
    fn equalization_lut_quantizes_to_requested_levels() {
        let cdf = [0u32, 1, 2, 3, 4];
        let out_levels = 2u32;
        let lut = equalization_lut(&cdf, 4, out_levels);
        // Only two levels: every entry must be exactly 0.0 or 1.0.
        let two_valued = lut.iter().all(|&v| close(v, 0.0) || close(v, 1.0));
        assert!(two_valued);
    }

    #[test]
    fn equalization_lut_out_levels_one_is_continuous() {
        let cdf = [0u32, 3, 6, 10];
        let lut = equalization_lut(&cdf, 10, 1);
        // Continuous normalization: the 6/10 relative to cdf_min=3 gives 3/7.
        let expected = 3.0f32 / 7.0f32;
        assert!(close(lut[2], expected));
    }

    #[test]
    fn apply_equalization_maps_into_range() {
        let cdf = [0u32, 2, 5, 9, 10];
        let lut = equalization_lut(&cdf, 10, 256);
        let out = apply_equalization(0.5, &lut, 0.0, 1.0);
        assert!((0.0..=1.0).contains(&out));
    }

    #[test]
    fn apply_equalization_empty_lut_is_passthrough() {
        let out = apply_equalization(0.42, &[], 0.0, 1.0);
        assert!(close(out, 0.42));
    }

    #[test]
    fn apply_equalization_degenerate_range_returns_min() {
        let lut = [0.0f32, 0.5, 1.0];
        let out = apply_equalization(7.0, &lut, 2.0, 2.0);
        assert!(close(out, 2.0));
    }

    #[test]
    fn apply_equalization_respects_nonzero_offset_range() {
        // A full-white LUT maps every sample to range_max.
        let lut = [1.0f32, 1.0, 1.0, 1.0];
        let out = apply_equalization(5.0, &lut, 10.0, 20.0);
        assert!(close(out, 20.0));
    }

    #[test]
    fn apply_equalization_is_monotonic_across_samples() {
        let hist = build_histogram(&[0.0, 0.3, 0.3, 0.6, 1.0], 8, 0.0, 1.0);
        let total = total_count(&hist);
        let cdf = cumulative_distribution(&hist);
        let lut = equalization_lut(&cdf, total, 256);
        let lo = apply_equalization(0.1, &lut, 0.0, 1.0);
        let mid = apply_equalization(0.5, &lut, 0.0, 1.0);
        let hi = apply_equalization(0.9, &lut, 0.0, 1.0);
        assert!(mid >= lo - TEST_EPS);
        assert!(hi >= mid - TEST_EPS);
    }

    #[test]
    fn clip_histogram_caps_tall_bins() {
        let mut hist = [10u32, 1, 1, 1];
        clip_histogram(&mut hist, 4);
        let peak_capped = hist[0] <= 4 + total_count(&[6]);
        assert!(peak_capped);
        assert!(hist[0] >= 4);
    }

    #[test]
    fn clip_histogram_conserves_total_mass() {
        let mut hist = [10u32, 2, 0, 0, 8];
        let before = total_count(&hist);
        clip_histogram(&mut hist, 3);
        let after = total_count(&hist);
        assert_eq!(after, before);
    }

    #[test]
    fn clip_histogram_redistributes_remainder_to_leading_bins() {
        // surplus = (7-2) = 5 over 4 bins -> per_bin 1, remainder 1.
        let mut hist = [7u32, 0, 0, 0];
        clip_histogram(&mut hist, 2);
        // bin0: capped to 2, +1 (per_bin) +1 (remainder) = 4.
        assert_eq!(hist[0], 4);
        // bin1..: 0 + per_bin(1) = 1 each.
        assert_eq!(hist[1], 1);
        assert_eq!(hist[2], 1);
        assert_eq!(hist[3], 1);
    }

    #[test]
    fn clip_histogram_empty_is_noop() {
        let mut hist: [u32; 0] = [];
        clip_histogram(&mut hist, 5);
        assert_eq!(hist.len(), 0);
    }

    #[test]
    fn clip_histogram_no_surplus_leaves_bins_unchanged() {
        let mut hist = [1u32, 2, 3];
        let before = hist;
        clip_histogram(&mut hist, 100);
        assert_eq!(hist, before);
    }

    #[test]
    fn gpu_storage_bytes_matches_u32_stride() {
        let bins = 16usize;
        assert_eq!(gpu_storage_bytes(bins), storage_bytes(U32_STRIDE, bins));
    }

    #[test]
    fn gpu_storage_bytes_empty_reserves_one_element() {
        assert_eq!(gpu_storage_bytes(0), U32_STRIDE);
    }

    #[test]
    fn end_to_end_equalization_spreads_a_peaked_distribution() {
        // A distribution concentrated in the low bins should map its dense
        // region toward higher output levels after equalization.
        let mut samples: Vec<f32> = Vec::new();
        samples.extend(core::iter::repeat_n(0.05f32, 90));
        samples.extend(core::iter::repeat_n(0.95f32, 10));
        let bins = 16usize;
        let hist = build_histogram(&samples, bins, 0.0, 1.0);
        let total = total_count(&hist);
        let cdf = cumulative_distribution(&hist);
        let lut = equalization_lut(&cdf, total, 256);
        let out_low = apply_equalization(0.05, &lut, 0.0, 1.0);
        let out_high = apply_equalization(0.95, &lut, 0.0, 1.0);
        assert!(out_high >= out_low - TEST_EPS);
        assert!(close(out_high, 1.0));
    }
}
