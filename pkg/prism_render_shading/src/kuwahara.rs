//! Backend-neutral CPU golden for the classic four-quadrant Kuwahara filter.
//!
//! Kuwahara is an edge-preserving, painterly smoothing operator. Around each
//! pixel a square neighbourhood of radius `r` is split into four overlapping
//! quadrants — top-left, top-right, bottom-left, bottom-right — each an
//! `(r + 1) x (r + 1)` block sharing the centre pixel. For every quadrant we
//! take the RGB mean and the *variance of its luminance*; the output pixel is
//! the mean of the quadrant with the smallest luminance variance. Choosing the
//! flattest (lowest-variance) region is what keeps hard edges crisp while
//! flooding smooth interiors with a single averaged colour, giving the
//! characteristic oil-painting look.
//!
//! Luminance uses the Rec. 709 weights `(0.2126, 0.7152, 0.0722)` on linear
//! RGB, matching the rest of the shading stack. Variance is the mean squared
//! deviation of per-sample luminance about the quadrant mean's luminance; no
//! square root is needed because only the *ordering* of variances matters for
//! the min-variance selection, so the whole module stays free of transcendental
//! or power functions. Every function is mirrored arm-for-arm — same constants,
//! same quadrant-selection order — by `shaders/kuwahara.wesl` so the CPU golden
//! and the GPU twin agree.

use bevy_math::Vec3;

/// Rec. 709 luma weights (linear `sRGB` primaries) used for the variance metric.
pub const KUWAHARA_LUMA_WEIGHTS: Vec3 = Vec3::new(0.2126, 0.7152, 0.0722);

/// Rec. 709 relative luminance of a linear RGB sample.
///
/// This is the scalar the quadrant variance is computed over; it weights the
/// green channel most heavily, matching human sensitivity.
#[must_use]
pub fn luminance(c: Vec3) -> f32 {
    c.x * KUWAHARA_LUMA_WEIGHTS.x + c.y * KUWAHARA_LUMA_WEIGHTS.y + c.z * KUWAHARA_LUMA_WEIGHTS.z
}

/// Arithmetic mean of a quadrant's RGB samples.
///
/// An empty slice has no colour to average, so it returns [`Vec3::ZERO`]; this
/// is the neutral element consumed by [`select_min_variance`] when a quadrant is
/// degenerate.
#[must_use]
pub fn region_mean(samples: &[Vec3]) -> Vec3 {
    if samples.is_empty() {
        return Vec3::ZERO;
    }
    let mut sum = Vec3::ZERO;
    for &s in samples {
        sum += s;
    }
    sum / samples.len() as f32
}

/// Mean squared deviation of the samples' luminance about `mean`'s luminance.
///
/// `mean` is the quadrant's RGB mean (see [`region_mean`]); its luminance is the
/// reference point. Because only the relative ordering of quadrant variances
/// drives the selection, this returns the *variance* (no square root). An empty
/// or single-sample quadrant has no spread and returns `0.0`.
#[must_use]
pub fn region_luma_variance(samples: &[Vec3], mean: Vec3) -> f32 {
    if samples.len() < 2 {
        return 0.0;
    }
    let mean_luma = luminance(mean);
    let mut acc = 0.0;
    for &s in samples {
        let d = luminance(s) - mean_luma;
        acc += d * d;
    }
    acc / samples.len() as f32
}

/// Return the RGB mean of the quadrant with the smallest luminance variance.
///
/// `regions` pairs each quadrant's `(mean, variance)`. Ties keep the *first*
/// minimum in iteration order (deterministic, matching the WESL twin's forward
/// scan). An empty list returns [`Vec3::ZERO`].
#[must_use]
pub fn select_min_variance(regions: &[(Vec3, f32)]) -> Vec3 {
    let mut best: Option<(Vec3, f32)> = None;
    for &(mean, var) in regions {
        match best {
            None => best = Some((mean, var)),
            Some((_, best_var)) if var < best_var => best = Some((mean, var)),
            _ => {}
        }
    }
    best.map_or(Vec3::ZERO, |(mean, _)| mean)
}

/// Painterly Kuwahara controls. `Default` is disabled with radius 2, so the
/// filter is the identity until a caller opts in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KuwaharaParams {
    /// Neighbourhood radius `r`; each of the four quadrants is `(r + 1)^2`.
    pub radius: u32,
    /// Master enable. When `false` the filter passes the centre pixel through.
    pub enabled: bool,
}

impl Default for KuwaharaParams {
    fn default() -> Self {
        Self {
            radius: 2,
            enabled: false,
        }
    }
}

/// Apply the Kuwahara selection to precomputed quadrant statistics.
///
/// When the filter is disabled, or there are no quadrants to choose from, the
/// `center` pixel is returned unchanged (identity). Otherwise the mean of the
/// lowest-variance quadrant is returned via [`select_min_variance`]. Mirrors the
/// WESL twin arm-for-arm.
#[must_use]
pub fn apply_kuwahara(regions: &[(Vec3, f32)], center: Vec3, params: &KuwaharaParams) -> Vec3 {
    if !params.enabled || regions.is_empty() {
        return center;
    }
    select_min_variance(regions)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() < 1.0e-5, "{a} != {b}");
    }

    fn approx3(a: Vec3, b: Vec3) {
        approx(a.x, b.x);
        approx(a.y, b.y);
        approx(a.z, b.z);
    }

    // --- luminance ---

    #[test]
    fn luminance_weights_sum_to_one_on_white() {
        // Rec. 709 weights partition unity, so white maps to luminance 1.
        approx(luminance(Vec3::ONE), 1.0);
    }

    #[test]
    fn luminance_isolates_channels() {
        approx(luminance(Vec3::new(1.0, 0.0, 0.0)), 0.2126);
        approx(luminance(Vec3::new(0.0, 1.0, 0.0)), 0.7152);
        approx(luminance(Vec3::new(0.0, 0.0, 1.0)), 0.0722);
    }

    #[test]
    fn luminance_of_grey_is_scalar() {
        // A neutral grey has luminance equal to its (single) channel value.
        approx(luminance(Vec3::splat(0.5)), 0.5);
    }

    // --- region_mean ---

    #[test]
    fn region_mean_averages_samples() {
        let samples = [Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.5, 0.25)];
        approx3(region_mean(&samples), Vec3::new(0.5, 0.25, 0.125));
    }

    #[test]
    fn region_mean_single_sample_is_that_sample() {
        let s = Vec3::new(0.3, 0.6, 0.9);
        approx3(region_mean(&[s]), s);
    }

    #[test]
    fn region_mean_empty_is_zero() {
        approx3(region_mean(&[]), Vec3::ZERO);
    }

    // --- region_luma_variance ---

    #[test]
    fn region_luma_variance_uniform_is_zero() {
        // A flat quadrant has no luminance spread.
        let samples = [Vec3::splat(0.4); 4];
        let mean = region_mean(&samples);
        approx(region_luma_variance(&samples, mean), 0.0);
    }

    #[test]
    fn region_luma_variance_known_value() {
        // Grey 0.0 and grey 1.0: mean luma 0.5, deviations +/-0.5,
        // variance = (0.25 + 0.25) / 2 = 0.25.
        let samples = [Vec3::ZERO, Vec3::ONE];
        let mean = region_mean(&samples);
        approx(region_luma_variance(&samples, mean), 0.25);
    }

    #[test]
    fn region_luma_variance_empty_is_zero() {
        approx(region_luma_variance(&[], Vec3::ZERO), 0.0);
    }

    #[test]
    fn region_luma_variance_single_is_zero() {
        approx(
            region_luma_variance(&[Vec3::new(0.2, 0.7, 0.1)], Vec3::new(0.2, 0.7, 0.1)),
            0.0,
        );
    }

    // --- select_min_variance ---

    #[test]
    fn select_min_variance_picks_lowest() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        let c = Vec3::new(0.0, 0.0, 1.0);
        let regions = [(a, 0.9), (b, 0.1), (c, 0.5)];
        approx3(select_min_variance(&regions), b);
    }

    #[test]
    fn select_min_variance_tie_takes_first() {
        let a = Vec3::new(0.1, 0.2, 0.3);
        let b = Vec3::new(0.4, 0.5, 0.6);
        let regions = [(a, 0.25), (b, 0.25)];
        approx3(select_min_variance(&regions), a);
    }

    #[test]
    fn select_min_variance_empty_is_zero() {
        approx3(select_min_variance(&[]), Vec3::ZERO);
    }

    // --- params ---

    #[test]
    fn default_params_are_disabled_radius_two() {
        let p = KuwaharaParams::default();
        assert_eq!(p.radius, 2);
        assert!(!p.enabled);
    }

    // --- apply_kuwahara ---

    #[test]
    fn apply_disabled_is_identity() {
        let center = Vec3::new(0.7, 0.2, 0.4);
        let regions = [(Vec3::ZERO, 0.0), (Vec3::ONE, 0.5)];
        let p = KuwaharaParams {
            radius: 2,
            enabled: false,
        };
        approx3(apply_kuwahara(&regions, center, &p), center);
    }

    #[test]
    fn apply_empty_regions_returns_center() {
        let center = Vec3::new(0.7, 0.2, 0.4);
        let p = KuwaharaParams {
            radius: 3,
            enabled: true,
        };
        approx3(apply_kuwahara(&[], center, &p), center);
    }

    #[test]
    fn apply_uniform_region_is_identity() {
        // All quadrants equal the centre with zero variance, so the selection
        // returns the centre colour: a flat image is a fixed point.
        let center = Vec3::new(0.3, 0.5, 0.8);
        let regions = [(center, 0.0), (center, 0.0), (center, 0.0), (center, 0.0)];
        let p = KuwaharaParams {
            radius: 2,
            enabled: true,
        };
        approx3(apply_kuwahara(&regions, center, &p), center);
    }

    #[test]
    fn apply_enabled_selects_min_variance_quadrant() {
        let center = Vec3::new(0.5, 0.5, 0.5);
        let flat = Vec3::new(0.2, 0.3, 0.4);
        let noisy = Vec3::new(0.9, 0.1, 0.0);
        let regions = [(noisy, 0.8), (flat, 0.05), (noisy, 0.6)];
        let p = KuwaharaParams {
            radius: 2,
            enabled: true,
        };
        approx3(apply_kuwahara(&regions, center, &p), flat);
    }
}
