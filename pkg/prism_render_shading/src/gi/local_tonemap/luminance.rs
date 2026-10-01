//! Luminance extraction, log-luminance encoding, zonal statistics, and
//! colour-preserving luminance re-application — backend-neutral CPU golden.
//!
//! Local (zonal) tone mapping operates almost entirely in the scalar luminance
//! domain: it extracts a single perceptual-brightness channel from linear HDR
//! radiance, estimates a *local* adaptation level around every pixel, compresses
//! that level, and finally scales the original RGB back by the per-pixel ratio
//! `Lout / Lin` so that hue and saturation survive the compression. This module
//! owns every piece of that scalar bookkeeping:
//!
//! * **Rec. 709 luminance** — `dot(rgb, (0.2126, 0.7152, 0.0722))`, the linear
//!   sRGB relative-luminance weights shared by the rest of the GI tail.
//! * **Log-luminance** — a numerically guarded natural-log encode (and its
//!   inverse) used by the bilateral base/detail split, where multiplicative
//!   contrast becomes additive and compression becomes a simple affine scale.
//! * **Zonal statistics** — mean, min, max, and log-average (geometric mean)
//!   over an arbitrary luminance window plus a fixed-range histogram, the
//!   building blocks a tile/zone adaptation estimator consumes.
//! * **Colour-preserving re-application** — `apply_luminance_ratio`, which maps
//!   a tone-mapped luminance back onto RGB through a clamped ratio.
//!
//! # Conventions
//! * Deterministic pure functions, no RNG / IO / GPU / unsafe / global state.
//! * Transcendental math via [`bevy_math::ops`]; `sqrt` via the inherent method.
//! * Every log takes a floored, strictly-positive argument; every ratio and
//!   output is finite and clamped. No path can emit `NaN` or `inf`.
//! * Windows are passed as borrowed `&[f32]` slices; nothing here allocates.
//!
//! # References
//! * Durand & Dorsey, "Fast Bilateral Filtering for the Display of
//!   High-Dynamic-Range Images", SIGGRAPH 2002 — log-domain luminance split.
//! * Reinhard et al., "Photographic Tone Reproduction for Digital Images",
//!   SIGGRAPH 2002 — log-average luminance key estimation.
//! * ITU-R BT.709 — relative luminance primary weights.

use bevy_math::ops;

/// Rec. 709 relative-luminance weights for linear sRGB primaries.
pub const LUMINANCE_WEIGHTS: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// Floor applied before any logarithm so the argument stays strictly positive.
pub const LUMINANCE_EPSILON: f32 = 1.0e-6;

/// Rec. 709 relative luminance of a linear RGB radiance sample.
///
/// Negative channels (which can appear transiently in wide-gamut working
/// spaces) are floored to zero so the returned luminance is non-negative.
#[must_use]
pub fn luminance(rgb: [f32; 3]) -> f32 {
    let r = rgb[0].max(0.0);
    let g = rgb[1].max(0.0);
    let b = rgb[2].max(0.0);
    r * LUMINANCE_WEIGHTS[0] + g * LUMINANCE_WEIGHTS[1] + b * LUMINANCE_WEIGHTS[2]
}

/// Natural-log encode of a luminance value, guarded by [`LUMINANCE_EPSILON`].
///
/// The argument is floored so black pixels encode to `ln(epsilon)` rather than
/// `-inf`; the result is therefore always finite.
#[must_use]
pub fn log_luminance(l: f32) -> f32 {
    ops::ln(l.max(LUMINANCE_EPSILON))
}

/// Inverse of [`log_luminance`]: exponentiate a log-domain luminance back to the
/// linear domain and floor to zero.
#[must_use]
pub fn exp_luminance(log_l: f32) -> f32 {
    let v = ops::exp(log_l);
    if v.is_finite() { v.max(0.0) } else { 0.0 }
}

/// Aggregate statistics over a luminance window.
///
/// All fields are finite. For an empty window the mean/min/max collapse to zero
/// and `log_mean` to [`LUMINANCE_EPSILON`] so downstream divisions stay safe.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZoneStats {
    /// Arithmetic mean of the (non-negative) luminances.
    pub mean: f32,
    /// Smallest luminance in the window.
    pub min: f32,
    /// Largest luminance in the window.
    pub max: f32,
    /// Geometric mean, i.e. `exp(mean(log L))` — the Reinhard key estimate.
    pub log_mean: f32,
    /// Number of samples that contributed.
    pub count: usize,
}

impl Default for ZoneStats {
    fn default() -> Self {
        Self {
            mean: 0.0,
            min: 0.0,
            max: 0.0,
            log_mean: LUMINANCE_EPSILON,
            count: 0,
        }
    }
}

/// Compute [`ZoneStats`] over a window of luminance samples.
///
/// Samples are floored to zero before aggregation. An empty slice yields the
/// [`ZoneStats::default`] (all-safe) value.
#[must_use]
pub fn zone_stats(window: &[f32]) -> ZoneStats {
    if window.is_empty() {
        return ZoneStats::default();
    }
    let mut sum = 0.0_f32;
    let mut log_sum = 0.0_f32;
    let mut min = f32::INFINITY;
    let mut max = 0.0_f32;
    for &raw in window {
        let l = if raw.is_finite() { raw.max(0.0) } else { 0.0 };
        sum += l;
        log_sum += log_luminance(l);
        if l < min {
            min = l;
        }
        if l > max {
            max = l;
        }
    }
    let count = window.len();
    let inv = 1.0 / count as f32;
    let min = if min.is_finite() { min } else { 0.0 };
    ZoneStats {
        mean: sum * inv,
        min,
        max,
        log_mean: exp_luminance(log_sum * inv).max(LUMINANCE_EPSILON),
        count,
    }
}

/// Arithmetic mean of a luminance window (non-negative, finite).
#[must_use]
pub fn zone_mean(window: &[f32]) -> f32 {
    zone_stats(window).mean
}

/// Log-average (geometric mean) luminance — the Reinhard key.
#[must_use]
pub fn zone_log_mean(window: &[f32]) -> f32 {
    zone_stats(window).log_mean
}

/// A fixed-range luminance histogram over the *log* domain.
///
/// Bins span `[log(min_l), log(max_l)]` uniformly; values below/above the range
/// clamp into the first/last bin. The histogram is a core input for
/// contrast-aware zonal adaptation (where the median or a percentile bin picks
/// the local key).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogHistogramRange {
    /// Lower luminance bound (floored to [`LUMINANCE_EPSILON`]).
    pub min_l: f32,
    /// Upper luminance bound (kept strictly above `min_l`).
    pub max_l: f32,
}

impl Default for LogHistogramRange {
    fn default() -> Self {
        Self {
            min_l: 1.0e-3,
            max_l: 1.0e4,
        }
    }
}

impl LogHistogramRange {
    /// Return a sanitised `(log_min, log_max)` pair with `log_max > log_min`.
    #[must_use]
    fn log_bounds(self) -> (f32, f32) {
        let lo = self.min_l.max(LUMINANCE_EPSILON);
        let hi = self.max_l.max(lo * (1.0 + 1.0e-3) + LUMINANCE_EPSILON);
        (ops::ln(lo), ops::ln(hi))
    }

    /// Map a luminance to a bin index in `[0, bin_count)`.
    ///
    /// A `bin_count` of zero returns zero (there is no valid bin to pick).
    #[must_use]
    pub fn bin_of(self, l: f32, bin_count: usize) -> usize {
        if bin_count == 0 {
            return 0;
        }
        let (log_min, log_max) = self.log_bounds();
        let log_l = log_luminance(l);
        let t = ((log_l - log_min) / (log_max - log_min)).clamp(0.0, 1.0);
        let idx = (t * bin_count as f32) as usize;
        idx.min(bin_count - 1)
    }

    /// Luminance at the centre of bin `bin` for a histogram of `bin_count` bins.
    #[must_use]
    pub fn bin_center(self, bin: usize, bin_count: usize) -> f32 {
        if bin_count == 0 {
            return self.min_l.max(LUMINANCE_EPSILON);
        }
        let (log_min, log_max) = self.log_bounds();
        let b = bin.min(bin_count - 1);
        let t = (b as f32 + 0.5) / bin_count as f32;
        exp_luminance(log_min + t * (log_max - log_min))
    }
}

/// Accumulate a window of luminances into `bins` using `range` (log domain).
///
/// `bins` is cleared to zero first. A zero-length `bins` slice is a no-op.
pub fn accumulate_log_histogram(window: &[f32], range: LogHistogramRange, bins: &mut [u32]) {
    for b in bins.iter_mut() {
        *b = 0;
    }
    let bin_count = bins.len();
    if bin_count == 0 {
        return;
    }
    for &raw in window {
        let l = if raw.is_finite() { raw.max(0.0) } else { 0.0 };
        let idx = range.bin_of(l, bin_count);
        bins[idx] = bins[idx].saturating_add(1);
    }
}

/// Return the luminance at the centre of the bin containing the cumulative
/// `percentile` (`0..=1`) of the histogram mass.
///
/// Useful to pick a robust local key (e.g. the median at `0.5`) that ignores
/// fireflies. An empty histogram returns the range's lower bound.
#[must_use]
pub fn histogram_percentile_luminance(
    bins: &[u32],
    range: LogHistogramRange,
    percentile: f32,
) -> f32 {
    let bin_count = bins.len();
    if bin_count == 0 {
        return range.min_l.max(LUMINANCE_EPSILON);
    }
    let total: u64 = bins.iter().map(|&c| u64::from(c)).sum();
    if total == 0 {
        return range.bin_center(bin_count / 2, bin_count);
    }
    let p = percentile.clamp(0.0, 1.0);
    let target = (p * total as f32).max(1.0);
    let mut running = 0.0_f32;
    for (i, &c) in bins.iter().enumerate() {
        running += c as f32;
        if running >= target {
            return range.bin_center(i, bin_count);
        }
    }
    range.bin_center(bin_count - 1, bin_count)
}

/// Scale `rgb` so its luminance becomes `l_out` while preserving chromaticity.
///
/// The ratio `l_out / l_in` is clamped to `[0, max_ratio]` to tame fireflies and
/// division blow-ups. When the input luminance is negligible the pixel is
/// treated as achromatic and the output is a neutral grey at `l_out`.
#[must_use]
pub fn apply_luminance_ratio(rgb: [f32; 3], l_in: f32, l_out: f32, max_ratio: f32) -> [f32; 3] {
    let l_out = if l_out.is_finite() { l_out.max(0.0) } else { 0.0 };
    let cap = if max_ratio.is_finite() && max_ratio > 0.0 {
        max_ratio
    } else {
        f32::MAX
    };
    if l_in <= LUMINANCE_EPSILON {
        return [l_out, l_out, l_out];
    }
    let ratio = (l_out / l_in).clamp(0.0, cap);
    let mut out = [0.0_f32; 3];
    for (o, &c) in out.iter_mut().zip(rgb.iter()) {
        let v = c.max(0.0) * ratio;
        *o = if v.is_finite() { v } else { 0.0 };
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() < 1.0e-4, "{a} != {b}");
    }

    #[test]
    fn luminance_weights_sum_to_one() {
        approx(
            LUMINANCE_WEIGHTS[0] + LUMINANCE_WEIGHTS[1] + LUMINANCE_WEIGHTS[2],
            1.0,
        );
    }

    #[test]
    fn luminance_of_white_is_one() {
        approx(luminance([1.0, 1.0, 1.0]), 1.0);
    }

    #[test]
    fn luminance_floors_negatives() {
        approx(luminance([-5.0, 0.0, 0.0]), 0.0);
    }

    #[test]
    fn log_exp_round_trip() {
        for &l in &[1.0e-3_f32, 0.18, 1.0, 42.0, 5000.0] {
            let got = exp_luminance(log_luminance(l));
            // Relative tolerance: `exp(ln(x))` loses a few ULPs at large `x`.
            assert!((got - l).abs() <= 1.0e-3 * l.max(1.0), "{got} != {l}");
        }
    }

    #[test]
    fn log_luminance_of_black_is_finite() {
        assert!(log_luminance(0.0).is_finite());
        assert!(exp_luminance(log_luminance(0.0)).is_finite());
    }

    #[test]
    fn exp_luminance_handles_overflow() {
        let v = exp_luminance(1.0e9);
        assert!(v.is_finite() || v == 0.0);
        assert!(exp_luminance(1.0e9) >= 0.0);
    }

    #[test]
    fn empty_window_is_safe() {
        let s = zone_stats(&[]);
        assert_eq!(s.count, 0);
        approx(s.mean, 0.0);
        assert!(s.log_mean >= LUMINANCE_EPSILON);
    }

    #[test]
    fn zone_stats_basic() {
        let w = [1.0_f32, 2.0, 3.0, 4.0];
        let s = zone_stats(&w);
        approx(s.mean, 2.5);
        approx(s.min, 1.0);
        approx(s.max, 4.0);
        assert_eq!(s.count, 4);
        // Geometric mean < arithmetic mean.
        assert!(s.log_mean < s.mean);
    }

    #[test]
    fn zone_stats_floors_non_finite() {
        let w = [f32::NAN, -1.0, 2.0];
        let s = zone_stats(&w);
        assert!(s.mean.is_finite());
        approx(s.min, 0.0);
        approx(s.max, 2.0);
    }

    #[test]
    fn log_mean_equals_value_for_constant_window() {
        let w = [3.0_f32; 8];
        approx(zone_log_mean(&w), 3.0);
    }

    #[test]
    fn histogram_bins_monotonic() {
        let range = LogHistogramRange::default();
        let lo = range.bin_of(1.0e-3, 64);
        let mid = range.bin_of(1.0, 64);
        let hi = range.bin_of(1.0e4, 64);
        assert!(lo <= mid && mid <= hi);
        assert!(hi < 64);
    }

    #[test]
    fn histogram_bin_center_round_trips_bin() {
        let range = LogHistogramRange::default();
        for bin in [0usize, 10, 31, 63] {
            let c = range.bin_center(bin, 64);
            assert_eq!(range.bin_of(c, 64), bin);
        }
    }

    #[test]
    fn histogram_percentile_recovers_median() {
        let range = LogHistogramRange::default();
        // All mass at a single luminance -> any percentile returns its bin.
        let w = [1.0_f32; 100];
        let mut bins = [0u32; 64];
        accumulate_log_histogram(&w, range, &mut bins);
        let total: u32 = bins.iter().sum();
        assert_eq!(total, 100);
        let median = histogram_percentile_luminance(&bins, range, 0.5);
        approx(median, range.bin_center(range.bin_of(1.0, 64), 64));
    }

    #[test]
    fn histogram_empty_mass_returns_center() {
        let range = LogHistogramRange::default();
        let bins = [0u32; 32];
        let v = histogram_percentile_luminance(&bins, range, 0.5);
        assert!(v.is_finite() && v > 0.0);
    }

    #[test]
    fn ratio_preserves_hue() {
        let rgb = [0.2, 0.4, 0.8];
        let l_in = luminance(rgb);
        let out = apply_luminance_ratio(rgb, l_in, l_in * 0.5, 8.0);
        // Ratios between channels unchanged.
        approx(out[1] / out[0], rgb[1] / rgb[0]);
        approx(out[2] / out[0], rgb[2] / rgb[0]);
        approx(luminance(out), l_in * 0.5);
    }

    #[test]
    fn ratio_clamps_fireflies() {
        let rgb = [1.0, 1.0, 1.0];
        let out = apply_luminance_ratio(rgb, 1.0e-5, 10.0, 4.0);
        // Capped at max_ratio -> 4x, not 1e6x.
        approx(out[0], 4.0);
    }

    #[test]
    fn ratio_zero_input_is_grey() {
        let out = apply_luminance_ratio([0.0, 0.0, 0.0], 0.0, 0.3, 8.0);
        approx(out[0], 0.3);
        approx(out[1], 0.3);
        approx(out[2], 0.3);
    }

    #[test]
    fn ratio_never_emits_non_finite() {
        let out = apply_luminance_ratio([f32::INFINITY, 1.0, 1.0], 1.0, 1.0, f32::INFINITY);
        for c in out {
            assert!(c.is_finite());
        }
    }
}
