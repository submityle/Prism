//! Firefly (high-variance outlier) suppression — CPU golden.
//!
//! Path-traced and ReSTIR-style global illumination estimators occasionally
//! return a single pixel whose radiance is orders of magnitude brighter than
//! its neighbours — a *firefly*.  These are not scene features but Monte-Carlo
//! variance: a low-probability path (a near-specular bounce into a bright
//! source, say) that was sampled once and divided by a tiny PDF.  Left alone
//! they survive temporal accumulation and smear into persistent sparkles.
//!
//! This module is the backend-neutral, CPU-golden reference for the two
//! firefly-reduction primitives the GPU denoiser mirrors:
//!
//! * A **Karis-weighted mean** ("partial Karis average"), which folds a small
//!   neighbourhood of RGB samples together with the weight `w = 1 / (1 + luma)`
//!   so that bright outliers contribute proportionally less energy to the
//!   result.  This is the standard temporal-AA / firefly-reduction weighting
//!   introduced by Brian Karis (SIGGRAPH 2014, *High Quality Temporal
//!   Supersampling*).
//! * A **neighbourhood luminance clamp**, which rescales a center sample whose
//!   luminance exceeds a configurable multiple of the neighbourhood's reference
//!   luminance (mean or max).  Rescaling is applied uniformly across RGB so the
//!   sample's hue (chromaticity) is preserved while its brightness is pulled
//!   back into a plausible range.
//!
//! # Conventions
//! * Colours are linear RGB stored as `[f32; 3]`, matching the GPU twin.
//! * Luminance uses the Rec.709 / sRGB primaries `(0.2126, 0.7152, 0.0722)`.
//! * Every function is a deterministic pure function over slices/arrays; there
//!   is no RNG, IO, or GPU state, so CPU and GPU results are bit-comparable up
//!   to floating-point rounding.

/// Rec.709 (sRGB) luminance weight for the linear red channel.
const LUMA_R: f32 = 0.2126;
/// Rec.709 (sRGB) luminance weight for the linear green channel.
const LUMA_G: f32 = 0.7152;
/// Rec.709 (sRGB) luminance weight for the linear blue channel.
const LUMA_B: f32 = 0.0722;

/// Relative luminance of a linear-RGB colour under the Rec.709 primaries.
///
/// The weights sum to one, so a neutral colour `[v, v, v]` returns exactly `v`.
/// No clamping is performed; a negative input channel can yield a negative
/// result, which callers may treat as degenerate.
#[inline]
pub fn luminance(rgb: [f32; 3]) -> f32 {
    LUMA_R * rgb[0] + LUMA_G * rgb[1] + LUMA_B * rgb[2]
}

/// Karis anti-firefly weight for a given luminance, `w = 1 / (1 + luma)`.
///
/// The weight is monotonically decreasing in luminance: dark samples approach
/// a weight of one while very bright samples approach zero, which is exactly
/// what suppresses isolated bright outliers in a weighted average.  Negative
/// luminances are clamped to zero before weighting so the denominator stays at
/// or above one and the weight stays in `(0, 1]`.
#[inline]
pub fn karis_weight(luma: f32) -> f32 {
    1.0 / (1.0 + luma.max(0.0))
}

/// Averages a neighbourhood of linear-RGB samples using Karis anti-firefly
/// weights `w_i = 1 / (1 + luma_i)`.
///
/// The result is `sum_i(w_i * c_i) / sum_i(w_i)`: a tone-mapped-style weighted
/// mean that de-emphasises bright outliers while preserving the overall energy
/// of the darker, more reliable samples.  Because the weights are strictly
/// positive and the output is a convex combination of the inputs, the result's
/// luminance always lies between the minimum and maximum sample luminance.
///
/// An empty slice returns `[0.0; 3]`.
#[inline]
pub fn karis_weighted_mean(samples: &[[f32; 3]]) -> [f32; 3] {
    let mut sum = [0.0f32; 3];
    let mut weight_sum = 0.0f32;
    for &c in samples {
        let w = karis_weight(luminance(c));
        sum[0] += c[0] * w;
        sum[1] += c[1] * w;
        sum[2] += c[2] * w;
        weight_sum += w;
    }
    if weight_sum <= f32::MIN_POSITIVE {
        return [0.0; 3];
    }
    let inv = weight_sum.recip();
    [sum[0] * inv, sum[1] * inv, sum[2] * inv]
}

/// Which neighbourhood luminance statistic a firefly clamp is measured against.
///
/// Using the [`Mean`](NeighborhoodReference::Mean) is more aggressive — a lone
/// bright sample barely moves the mean, so the clamp threshold stays low — and
/// is the usual choice.  Using the [`Max`](NeighborhoodReference::Max) is more
/// conservative because the threshold rises to track the brightest legitimate
/// neighbour, clamping only samples that overshoot even that.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NeighborhoodReference {
    /// Compare against the arithmetic mean luminance of the neighbourhood.
    Mean,
    /// Compare against the maximum luminance of the neighbourhood.
    Max,
}

/// Mean Rec.709 luminance of a neighbourhood, or `0.0` for an empty slice.
#[inline]
pub fn mean_luminance(samples: &[[f32; 3]]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0f32;
    for &c in samples {
        sum += luminance(c);
    }
    sum / samples.len() as f32
}

/// Maximum Rec.709 luminance of a neighbourhood, or `0.0` for an empty slice.
#[inline]
pub fn max_luminance(samples: &[[f32; 3]]) -> f32 {
    let mut max = 0.0f32;
    for &c in samples {
        let l = luminance(c);
        if l > max {
            max = l;
        }
    }
    max
}

/// Clamps a center sample's luminance to `max_ratio` times a neighbourhood
/// reference luminance, preserving hue.
///
/// The threshold is `max_ratio * reference`, where `reference` is either the
/// mean or max neighbourhood luminance per `reference`.  If the center sample's
/// luminance exceeds the threshold, every channel is scaled uniformly by
/// `threshold / center_luma`, which pulls the brightness down to the threshold
/// while leaving chromaticity untouched.  Samples already within range (and the
/// degenerate cases below) are returned unchanged:
///
/// * `max_ratio <= 0` — a non-positive ratio has no meaningful threshold.
/// * `reference <= 0` — a dark/empty neighbourhood offers no scale to clamp to.
/// * `center_luma <= 0` — nothing to pull down.
///
/// This kills isolated fireflies while keeping the energy of in-range samples
/// exactly stable, matching the GPU clamp used before temporal accumulation.
#[inline]
pub fn clamp_firefly(
    center: [f32; 3],
    neighborhood: &[[f32; 3]],
    max_ratio: f32,
    reference: NeighborhoodReference,
) -> [f32; 3] {
    if max_ratio <= 0.0 {
        return center;
    }
    let reference_luma = match reference {
        NeighborhoodReference::Mean => mean_luminance(neighborhood),
        NeighborhoodReference::Max => max_luminance(neighborhood),
    };
    if reference_luma <= 0.0 {
        return center;
    }
    let center_luma = luminance(center);
    if center_luma <= 0.0 {
        return center;
    }
    let threshold = max_ratio * reference_luma;
    if center_luma <= threshold {
        return center;
    }
    let scale = threshold / center_luma;
    [center[0] * scale, center[1] * scale, center[2] * scale]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Luminance of a neutral grey equals its channel value, and the primary
    /// weights match Rec.709.
    #[test]
    fn luminance_matches_rec709() {
        assert!((luminance([0.5, 0.5, 0.5]) - 0.5).abs() < 1e-6);
        assert!((luminance([1.0, 0.0, 0.0]) - LUMA_R).abs() < 1e-6);
        assert!((luminance([0.0, 1.0, 0.0]) - LUMA_G).abs() < 1e-6);
        assert!((luminance([0.0, 0.0, 1.0]) - LUMA_B).abs() < 1e-6);
    }

    /// The Karis weight is one at zero luminance, decreasing and bounded in
    /// `(0, 1]`, and clamps negative luminance to the `luma = 0` case.
    #[test]
    fn karis_weight_is_monotone_decreasing() {
        assert!((karis_weight(0.0) - 1.0).abs() < 1e-6);
        assert!(karis_weight(1.0) < karis_weight(0.0));
        assert!(karis_weight(100.0) < karis_weight(1.0));
        assert!(karis_weight(100.0) > 0.0);
        // Negative luminance is clamped, so weight stays at the maximum.
        assert!((karis_weight(-5.0) - 1.0).abs() < 1e-6);
    }

    /// The Karis-weighted mean's luminance stays between the min and max of the
    /// sample luminances (it is a convex combination).
    #[test]
    fn karis_mean_luma_is_between_min_and_max() {
        let samples = [
            [0.1, 0.1, 0.1],
            [0.4, 0.2, 0.3],
            [50.0, 50.0, 50.0], // planted firefly
            [0.2, 0.25, 0.15],
        ];
        let mut min_l = f32::INFINITY;
        let mut max_l = f32::NEG_INFINITY;
        for &c in &samples {
            let l = luminance(c);
            min_l = min_l.min(l);
            max_l = max_l.max(l);
        }
        let mean = karis_weighted_mean(&samples);
        let mean_l = luminance(mean);
        assert!(mean_l >= min_l - 1e-5, "{mean_l} < {min_l}");
        assert!(mean_l <= max_l + 1e-5, "{mean_l} > {max_l}");
        // The firefly must be heavily down-weighted: the Karis mean should be
        // far closer to the dark cluster than a plain arithmetic mean.
        let naive_l = (min_l + max_l) * 0.5;
        assert!(mean_l < naive_l, "karis {mean_l} not below naive {naive_l}");
    }

    /// An empty neighbourhood averages to black.
    #[test]
    fn karis_mean_of_empty_is_zero() {
        assert_eq!(karis_weighted_mean(&[]), [0.0; 3]);
    }

    /// The clamp pulls a planted outlier down to the mean-based threshold and
    /// preserves its hue (channel ratios).
    #[test]
    fn clamp_reduces_planted_firefly_and_keeps_hue() {
        let neighborhood = [
            [0.20, 0.10, 0.05],
            [0.22, 0.11, 0.06],
            [0.18, 0.09, 0.04],
            [0.21, 0.10, 0.05],
        ];
        let firefly = [40.0, 20.0, 10.0];
        let before = luminance(firefly);
        let clamped = clamp_firefly(firefly, &neighborhood, 4.0, NeighborhoodReference::Mean);
        let after = luminance(clamped);
        assert!(after < before, "clamp did not reduce luminance");

        let reference = mean_luminance(&neighborhood);
        let threshold = 4.0 * reference;
        assert!((after - threshold).abs() < 1e-4, "{after} vs {threshold}");

        // Hue preserved: the clamped colour is a positive scalar multiple of the
        // original, so channel ratios are unchanged.
        let scale = clamped[0] / firefly[0];
        for ch in 0..3 {
            assert!((clamped[ch] - firefly[ch] * scale).abs() < 1e-4);
        }
    }

    /// An in-range center sample passes through the clamp untouched under both
    /// reference modes.
    #[test]
    fn clamp_leaves_in_range_samples_unchanged() {
        let neighborhood = [
            [0.20, 0.10, 0.05],
            [0.22, 0.11, 0.06],
            [0.18, 0.09, 0.04],
        ];
        let center = [0.25, 0.12, 0.06];
        assert_eq!(
            clamp_firefly(center, &neighborhood, 8.0, NeighborhoodReference::Mean),
            center
        );
        assert_eq!(
            clamp_firefly(center, &neighborhood, 8.0, NeighborhoodReference::Max),
            center
        );
    }

    /// Degenerate parameters are no-ops rather than producing NaNs or zeros.
    #[test]
    fn clamp_degenerate_inputs_are_noops() {
        let neighborhood = [[0.2, 0.2, 0.2]];
        let center = [5.0, 5.0, 5.0];
        // Non-positive ratio.
        assert_eq!(
            clamp_firefly(center, &neighborhood, 0.0, NeighborhoodReference::Mean),
            center
        );
        // Dark/empty neighbourhood.
        assert_eq!(
            clamp_firefly(center, &[], 4.0, NeighborhoodReference::Mean),
            center
        );
        assert_eq!(
            clamp_firefly(center, &[[0.0, 0.0, 0.0]], 4.0, NeighborhoodReference::Max),
            center
        );
        // Black center.
        assert_eq!(
            clamp_firefly([0.0, 0.0, 0.0], &neighborhood, 4.0, NeighborhoodReference::Mean),
            [0.0, 0.0, 0.0]
        );
    }

    /// The max reference is more conservative than the mean: for the same ratio
    /// it clamps to a higher (or equal) threshold.
    #[test]
    fn max_reference_is_more_conservative_than_mean() {
        let neighborhood = [
            [0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1],
            [1.0, 1.0, 1.0],
        ];
        let firefly = [100.0, 100.0, 100.0];
        let by_mean = luminance(clamp_firefly(
            firefly,
            &neighborhood,
            2.0,
            NeighborhoodReference::Mean,
        ));
        let by_max = luminance(clamp_firefly(
            firefly,
            &neighborhood,
            2.0,
            NeighborhoodReference::Max,
        ));
        assert!(by_max >= by_mean, "max {by_max} < mean {by_mean}");
    }

    /// Repeated evaluation is bit-for-bit identical (determinism).
    #[test]
    fn firefly_helpers_are_deterministic() {
        let samples = [
            [0.3, 0.1, 0.7],
            [12.0, 3.0, 0.5],
            [0.05, 0.9, 0.2],
        ];
        let a = karis_weighted_mean(&samples);
        let b = karis_weighted_mean(&samples);
        assert_eq!(a, b);
        let c = clamp_firefly(samples[1], &samples, 3.0, NeighborhoodReference::Mean);
        let d = clamp_firefly(samples[1], &samples, 3.0, NeighborhoodReference::Mean);
        assert_eq!(c, d);
    }
}
