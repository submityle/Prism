//! Running mean/variance estimation (Welford) — CPU golden.
//!
//! ReLAX-style spatio-temporal denoisers steer their edge-stopping functions
//! and history clamps with a per-pixel estimate of luminance variance: regions
//! with high variance (noisy, under-converged, or disoccluded) get wider
//! filters and looser temporal clamps, while low-variance regions are left
//! crisp.  Estimating that variance *online*, as samples stream in, demands a
//! numerically stable one-pass algorithm — summing `x` and `x^2` separately
//! and subtracting loses catastrophic precision for large means.
//!
//! This module is the backend-neutral, CPU-golden reference for two primitives
//! the GPU denoiser mirrors:
//!
//! * [`WelfordAccumulator`] — Welford's online algorithm for the running mean
//!   and (population or sample) variance of a scalar stream, plus
//!   [`WelfordRgb`] which tracks the same statistics per RGB channel and for
//!   Rec.709 luminance to guide ReLAX.
//! * [`TemporalMoments`] — an exponential-moving-average pair of first and
//!   second raw moments `(m1, m2)` whose variance `m2 - m1^2` (clamped at zero)
//!   is the cheap temporal variance estimate used when per-sample history is
//!   unavailable.
//!
//! # Conventions
//! * Statistics are stored as `f32` to match the GPU twin, accepting the usual
//!   single-precision rounding.  Variance is clamped to be non-negative so
//!   floating-point cancellation can never feed a negative value into a square
//!   root downstream.
//! * Rec.709 luminance uses the primaries `(0.2126, 0.7152, 0.0722)`, matching
//!   the `firefly` sibling module.
//! * Every operation is deterministic and side-effect free (no RNG/IO/GPU).

use super::firefly::luminance;

/// Welford's online algorithm for the running mean and variance of a scalar
/// stream.
///
/// The accumulator maintains the sample count, the running mean, and `m2`, the
/// running sum of squared deviations from the current mean.  Each
/// [`push`](WelfordAccumulator::push) updates all three in one pass without
/// ever forming `sum(x^2)`, so it stays accurate even when the mean is large
/// relative to the spread.
///
/// Both the **population** variance (`m2 / n`) and the unbiased **sample**
/// variance (`m2 / (n - 1)`) are exposed; see
/// [`variance`](WelfordAccumulator::variance) and
/// [`sample_variance`](WelfordAccumulator::sample_variance).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WelfordAccumulator {
    /// Number of samples folded in so far.
    count: u64,
    /// Running arithmetic mean of the samples seen so far.
    mean: f32,
    /// Running sum of squared deviations from the mean (`sum (x_i - mean)^2`).
    m2: f32,
}

impl Default for WelfordAccumulator {
    fn default() -> Self {
        Self::EMPTY
    }
}

impl WelfordAccumulator {
    /// An accumulator holding no samples.
    pub const EMPTY: Self = Self {
        count: 0,
        mean: 0.0,
        m2: 0.0,
    };

    /// Folds a single scalar sample into the running statistics.
    ///
    /// Uses the canonical Welford update: the mean is nudged by
    /// `(x - mean) / n` and `m2` accumulates the product of the deviations
    /// taken before and after that nudge.
    #[inline]
    pub fn push(&mut self, x: f32) {
        self.count += 1;
        let delta = x - self.mean;
        self.mean += delta / self.count as f32;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }

    /// The number of samples folded in so far.
    #[inline]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// The running arithmetic mean, or `0.0` when no samples have been pushed.
    #[inline]
    pub fn mean(&self) -> f32 {
        self.mean
    }

    /// The **population** variance `m2 / n`.
    ///
    /// Returns `0.0` for an empty accumulator.  The result is clamped to be
    /// non-negative so rounding of `m2` can never yield a negative variance.
    #[inline]
    pub fn variance(&self) -> f32 {
        if self.count == 0 {
            return 0.0;
        }
        (self.m2 / self.count as f32).max(0.0)
    }

    /// The unbiased **sample** variance `m2 / (n - 1)` (Bessel's correction).
    ///
    /// Returns `0.0` when fewer than two samples have been pushed, since the
    /// correction is undefined there.  The result is clamped to be
    /// non-negative.
    #[inline]
    pub fn sample_variance(&self) -> f32 {
        if self.count < 2 {
            return 0.0;
        }
        (self.m2 / (self.count - 1) as f32).max(0.0)
    }

    /// The population standard deviation (`sqrt` of [`variance`]).
    ///
    /// [`variance`]: WelfordAccumulator::variance
    #[inline]
    pub fn std_dev(&self) -> f32 {
        self.variance().sqrt()
    }
}

/// Per-channel plus luminance Welford statistics for an RGB stream.
///
/// ReLAX guidance wants both the per-channel variance (to clamp history toward
/// each channel independently) and a scalar luminance variance (to drive the
/// edge-stopping function).  This tracks all four with a single pass over the
/// samples: three [`WelfordAccumulator`]s for the channels and one for Rec.709
/// luminance.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WelfordRgb {
    /// Per-channel accumulators in `[r, g, b]` order.
    channels: [WelfordAccumulator; 3],
    /// Accumulator over the Rec.709 luminance of each sample.
    luma: WelfordAccumulator,
}

impl WelfordRgb {
    /// An accumulator holding no samples.
    pub const EMPTY: Self = Self {
        channels: [WelfordAccumulator::EMPTY; 3],
        luma: WelfordAccumulator::EMPTY,
    };

    /// Folds a single linear-RGB sample into the per-channel and luminance
    /// statistics.
    #[inline]
    pub fn push(&mut self, rgb: [f32; 3]) {
        self.channels[0].push(rgb[0]);
        self.channels[1].push(rgb[1]);
        self.channels[2].push(rgb[2]);
        self.luma.push(luminance(rgb));
    }

    /// The number of samples folded in so far.
    #[inline]
    pub fn count(&self) -> u64 {
        self.luma.count()
    }

    /// The per-channel running mean in `[r, g, b]` order.
    #[inline]
    pub fn mean(&self) -> [f32; 3] {
        [
            self.channels[0].mean(),
            self.channels[1].mean(),
            self.channels[2].mean(),
        ]
    }

    /// The per-channel population variance in `[r, g, b]` order.
    #[inline]
    pub fn variance(&self) -> [f32; 3] {
        [
            self.channels[0].variance(),
            self.channels[1].variance(),
            self.channels[2].variance(),
        ]
    }

    /// The running mean of the samples' Rec.709 luminance.
    #[inline]
    pub fn luminance_mean(&self) -> f32 {
        self.luma.mean()
    }

    /// The population variance of the samples' Rec.709 luminance — the scalar
    /// signal ReLAX feeds to its edge-stopping function.
    #[inline]
    pub fn luminance_variance(&self) -> f32 {
        self.luma.variance()
    }
}

/// An exponentially-weighted pair of raw first and second moments used for a
/// cheap temporal variance estimate.
///
/// `m1` tracks `E[x]` and `m2` tracks `E[x^2]` under an exponential moving
/// average with blend factor `alpha`.  The variance is then `m2 - m1^2`,
/// clamped at zero.  This is the form the GPU denoiser stores in its history
/// texture: two floats per pixel from which variance is reconstructed on the
/// fly, avoiding a separate variance history.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TemporalMoments {
    /// First raw moment, an EMA estimate of `E[x]`.
    pub m1: f32,
    /// Second raw moment, an EMA estimate of `E[x^2]`.
    pub m2: f32,
}

impl TemporalMoments {
    /// Moments representing no accumulated history.
    pub const ZERO: Self = Self { m1: 0.0, m2: 0.0 };

    /// Seeds the moments from a single sample as if it were fully converged
    /// (`m1 = x`, `m2 = x^2`), giving a starting variance of zero.
    #[inline]
    pub fn from_sample(x: f32) -> Self {
        Self { m1: x, m2: x * x }
    }

    /// Blends a new sample into the moments with weight `alpha` and returns the
    /// updated moments.
    ///
    /// Each moment is an exponential moving average,
    /// `m <- lerp(m_prev, sample_moment, alpha)`, where `alpha` is clamped to
    /// `[0, 1]`.  An `alpha` of `0` ignores the new sample; `1` replaces the
    /// history with it.
    #[inline]
    pub fn blend(self, x: f32, alpha: f32) -> Self {
        let a = alpha.clamp(0.0, 1.0);
        Self {
            m1: self.m1 + (x - self.m1) * a,
            m2: self.m2 + (x * x - self.m2) * a,
        }
    }

    /// The temporal variance `m2 - m1^2`, clamped to be non-negative.
    ///
    /// Jensen's inequality guarantees `E[x^2] >= E[x]^2` in exact arithmetic,
    /// but `f32` cancellation can push the difference slightly negative; the
    /// clamp keeps the result safe for a downstream square root.
    #[inline]
    pub fn variance(self) -> f32 {
        (self.m2 - self.m1 * self.m1).max(0.0)
    }
}

/// Convenience wrapper around [`TemporalMoments::blend`]: given previous
/// moments, a new sample, and a blend `alpha`, returns the updated moments.
///
/// Provided as a free function so call sites that store moments in plain data
/// (mirroring a GPU history texel) can update them without a method receiver.
#[inline]
pub fn update_temporal_moments(prev: TemporalMoments, sample: f32, alpha: f32) -> TemporalMoments {
    prev.blend(sample, alpha)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Naive two-pass population mean/variance for cross-checking Welford.
    fn naive_mean_variance(data: &[f32]) -> (f32, f32) {
        let n = data.len() as f32;
        let mean = data.iter().copied().sum::<f32>() / n;
        let var = data.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n;
        (mean, var)
    }

    /// Welford's mean and population variance match the two-pass computation on
    /// a known dataset within tolerance.
    #[test]
    fn welford_matches_naive() {
        let data = [2.0f32, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let mut acc = WelfordAccumulator::EMPTY;
        for &x in &data {
            acc.push(x);
        }
        let (mean, var) = naive_mean_variance(&data);
        assert_eq!(acc.count(), data.len() as u64);
        assert!((acc.mean() - mean).abs() < 1e-5, "{} vs {mean}", acc.mean());
        // This dataset is the textbook one: mean 5, population variance 4.
        assert!((acc.mean() - 5.0).abs() < 1e-5);
        assert!((acc.variance() - 4.0).abs() < 1e-5);
        assert!((acc.variance() - var).abs() < 1e-5);
        // Sample variance uses n-1 = 7: 32 / 7.
        assert!((acc.sample_variance() - 32.0 / 7.0).abs() < 1e-5);
        assert!((acc.std_dev() - 2.0).abs() < 1e-5);
    }

    /// Welford stays accurate when the mean dwarfs the spread, where the naive
    /// `sum(x^2) - sum(x)^2/n` approach would lose precision.
    #[test]
    fn welford_is_stable_with_large_offset() {
        let base = 1.0e6f32;
        let data = [base, base + 1.0, base + 2.0, base + 3.0, base + 4.0];
        let mut acc = WelfordAccumulator::EMPTY;
        for &x in &data {
            acc.push(x);
        }
        // Values are 1e6..1e6+4: mean offset 2, population variance 2.
        assert!((acc.mean() - (base + 2.0)).abs() < 1.0);
        assert!((acc.variance() - 2.0).abs() < 0.5, "{}", acc.variance());
    }

    /// Empty and single-sample edge cases are well defined and never negative.
    #[test]
    fn welford_edge_cases() {
        let empty = WelfordAccumulator::EMPTY;
        assert_eq!(empty.count(), 0);
        assert_eq!(empty.mean(), 0.0);
        assert_eq!(empty.variance(), 0.0);
        assert_eq!(empty.sample_variance(), 0.0);

        let mut one = WelfordAccumulator::EMPTY;
        one.push(3.5);
        assert_eq!(one.count(), 1);
        assert_eq!(one.mean(), 3.5);
        assert_eq!(one.variance(), 0.0);
        assert_eq!(one.sample_variance(), 0.0);
    }

    /// Variance is never negative, even for a constant stream where `m2` may
    /// round to a tiny negative value.
    #[test]
    fn variance_is_never_negative() {
        let mut acc = WelfordAccumulator::EMPTY;
        for _ in 0..1000 {
            acc.push(0.1);
        }
        assert!(acc.variance() >= 0.0);
        assert!(acc.sample_variance() >= 0.0);
        assert!(acc.variance() < 1e-6);
    }

    /// The RGB accumulator tracks per-channel statistics and a luminance
    /// variance consistent with the channel means.
    #[test]
    fn welford_rgb_tracks_channels_and_luminance() {
        let samples = [
            [1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0],
            [2.0, 2.0, 2.0],
        ];
        let mut acc = WelfordRgb::EMPTY;
        for &s in &samples {
            acc.push(s);
        }
        assert_eq!(acc.count(), 3);
        let mean = acc.mean();
        assert!((mean[0] - 2.0).abs() < 1e-5);
        assert!((mean[1] - 2.0).abs() < 1e-5);
        assert!((mean[2] - 2.0).abs() < 1e-5);
        // Green is constant -> zero variance; red/blue vary identically.
        let var = acc.variance();
        assert!(var[1] < 1e-6, "green variance {}", var[1]);
        assert!(var[0] > 0.0 && var[2] > 0.0);
        assert!((var[0] - var[2]).abs() < 1e-5);
        // Luminance mean equals the luminance of the mean here (linear op).
        assert!((acc.luminance_mean() - luminance(mean)).abs() < 1e-5);
        assert!(acc.luminance_variance() >= 0.0);
    }

    /// Temporal moments: a constant stream converges to zero variance, and a
    /// two-point stream reproduces the closed-form EMA variance.
    #[test]
    fn temporal_moments_track_variance() {
        // Constant input -> variance collapses toward zero.
        let mut m = TemporalMoments::from_sample(5.0);
        for _ in 0..32 {
            m = update_temporal_moments(m, 5.0, 0.2);
        }
        assert!(m.variance() < 1e-4, "{}", m.variance());
        assert!((m.m1 - 5.0).abs() < 1e-4);

        // Start from zero history, blend one sample of 10 with alpha 0.5:
        // m1 = 5, m2 = 50, variance = 50 - 25 = 25.
        let seeded = update_temporal_moments(TemporalMoments::ZERO, 10.0, 0.5);
        assert!((seeded.m1 - 5.0).abs() < 1e-5);
        assert!((seeded.m2 - 50.0).abs() < 1e-5);
        assert!((seeded.variance() - 25.0).abs() < 1e-5);
    }

    /// Temporal variance is clamped non-negative and `alpha` is clamped to the
    /// unit interval.
    #[test]
    fn temporal_variance_clamps() {
        // Hand-built moments with m2 < m1^2 (impossible in exact arithmetic)
        // must still yield a non-negative variance.
        let bogus = TemporalMoments { m1: 2.0, m2: 1.0 };
        assert_eq!(bogus.variance(), 0.0);

        // alpha > 1 behaves like alpha = 1 (full replacement).
        let replaced = update_temporal_moments(TemporalMoments::from_sample(1.0), 7.0, 5.0);
        assert!((replaced.m1 - 7.0).abs() < 1e-5);
        assert!((replaced.m2 - 49.0).abs() < 1e-5);
        // alpha < 0 behaves like alpha = 0 (ignore the sample).
        let ignored = update_temporal_moments(TemporalMoments::from_sample(1.0), 7.0, -5.0);
        assert_eq!(ignored, TemporalMoments::from_sample(1.0));
    }

    /// Replaying the same stream yields identical results (determinism).
    #[test]
    fn welford_is_deterministic() {
        let data = [0.3f32, 1.7, 2.9, 0.1, 4.4, 2.2];
        let run = || {
            let mut acc = WelfordAccumulator::EMPTY;
            for &x in &data {
                acc.push(x);
            }
            acc
        };
        assert_eq!(run(), run());
    }
}
