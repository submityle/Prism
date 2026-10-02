//! Error metrics for validating a candidate render against the reference image.
//!
//! The whole point of this offline tracer is to be a *ground truth*: real-time
//! global-illumination paths (`ReSTIR`, surface cache, froxel volumetrics) are
//! correct insofar as they converge to the same image this integrator produces.
//! This module turns that comparison into numbers. Given a reference buffer and
//! a candidate buffer of equal shape it reports the classical image-difference
//! metrics used in rendering papers: mean squared error, its root, mean
//! absolute error, the maximum per-channel absolute error, and a
//! luminance-relative mean squared error that stays finite in black regions.
//!
//! All reductions accumulate in `f64` so a large image does not lose precision
//! to `f32` rounding; only the final results are returned as `f32`. The single
//! transcendental used is `sqrt` (for the root-mean-squared error), honouring
//! the crate's determinism policy.

use super::film::Film;
use super::Vec3;

/// A small luminance floor added to the denominator of the relative error so a
/// (near-)black reference pixel cannot produce a division by zero or `NaN`.
const RELATIVE_EPS: f64 = 1e-6;

/// `Rec. 709` luminance weights, used to collapse an RGB difference to a single
/// perceptual channel for the relative-error denominator.
const LUMA_R: f64 = 0.212_639;
/// Green `Rec. 709` luminance weight.
const LUMA_G: f64 = 0.715_169;
/// Blue `Rec. 709` luminance weight.
const LUMA_B: f64 = 0.072_192;

/// The bundle of difference metrics between a candidate and a reference image.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ErrorMetrics {
    /// Mean squared error, averaged over every channel of every pixel.
    pub mse: f32,
    /// Root mean squared error (`sqrt(mse)`).
    pub rmse: f32,
    /// Mean absolute error, averaged over every channel of every pixel.
    pub mae: f32,
    /// Largest absolute per-channel difference anywhere in the image.
    pub max_abs: f32,
    /// Mean squared error normalized by reference luminance (plus a small
    /// floor), so bright and dark regions contribute comparably.
    pub relative_mse: f32,
}

impl ErrorMetrics {
    /// The all-zero metrics, i.e. two identical (or empty) images.
    pub const ZERO: Self = Self {
        mse: 0.0,
        rmse: 0.0,
        mae: 0.0,
        max_abs: 0.0,
        relative_mse: 0.0,
    };
}

/// Compares two equal-length pixel buffers, returning [`None`] when the lengths
/// differ (the shapes are not comparable) and [`ErrorMetrics::ZERO`] for a pair
/// of empty buffers.
#[must_use]
pub fn compare_pixels(reference: &[Vec3], candidate: &[Vec3]) -> Option<ErrorMetrics> {
    if reference.len() != candidate.len() {
        return None;
    }
    if reference.is_empty() {
        return Some(ErrorMetrics::ZERO);
    }

    let mut sum_sq = 0.0f64;
    let mut sum_abs = 0.0f64;
    let mut max_abs = 0.0f64;
    let mut sum_rel = 0.0f64;

    for (r, c) in reference.iter().zip(candidate) {
        let dr = f64::from(c.x) - f64::from(r.x);
        let dg = f64::from(c.y) - f64::from(r.y);
        let db = f64::from(c.z) - f64::from(r.z);

        sum_sq += dr * dr + dg * dg + db * db;
        sum_abs += dr.abs() + dg.abs() + db.abs();
        max_abs = max_abs.max(dr.abs()).max(dg.abs()).max(db.abs());

        // Normalize the per-pixel squared error by the reference luminance so a
        // fixed absolute error weighs more in dark regions than in bright ones.
        let luma = LUMA_R * f64::from(r.x) + LUMA_G * f64::from(r.y) + LUMA_B * f64::from(r.z);
        let denom = luma.abs() + RELATIVE_EPS;
        sum_rel += (dr * dr + dg * dg + db * db) / denom;
    }

    let channels = (reference.len() as f64) * 3.0;
    let mse = sum_sq / channels;
    let mae = sum_abs / channels;
    let relative_mse = sum_rel / channels;

    Some(ErrorMetrics {
        mse: mse as f32,
        rmse: mse.sqrt() as f32,
        mae: mae as f32,
        max_abs: max_abs as f32,
        relative_mse: relative_mse as f32,
    })
}

/// Compares two [`Film`]s, returning [`None`] when their dimensions differ.
#[must_use]
pub fn compare_films(reference: &Film, candidate: &Film) -> Option<ErrorMetrics> {
    if reference.width() != candidate.width() || reference.height() != candidate.height() {
        return None;
    }
    compare_pixels(reference.pixels(), candidate.pixels())
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::*;

    fn pixels(data: &[[f32; 3]]) -> Vec<Vec3> {
        data.iter().map(|a| Vec3::from_array(*a)).collect()
    }

    #[test]
    fn identical_images_have_zero_error() {
        let img = pixels(&[[1.0, 0.5, 0.25], [0.1, 0.2, 0.3]]);
        let m = compare_pixels(&img, &img).expect("equal lengths");
        assert_eq!(m, ErrorMetrics::ZERO);
    }

    #[test]
    fn empty_buffers_are_zero() {
        let m = compare_pixels(&[], &[]).expect("both empty");
        assert_eq!(m, ErrorMetrics::ZERO);
    }

    #[test]
    fn mismatched_lengths_return_none() {
        let a = pixels(&[[0.0, 0.0, 0.0]]);
        let b = pixels(&[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]);
        assert!(compare_pixels(&a, &b).is_none());
    }

    #[test]
    fn known_difference_matches_closed_form() {
        // A single channel differs by 2 in one pixel; everything else matches.
        let reference = pixels(&[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
        let candidate = pixels(&[[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
        let m = compare_pixels(&reference, &candidate).expect("equal lengths");
        // 6 channels total; one squared diff of 4 → mse = 4/6.
        assert!((f64::from(m.mse) - 4.0 / 6.0).abs() < 1e-6);
        assert!((f64::from(m.rmse) - (4.0f64 / 6.0).sqrt()).abs() < 1e-6);
        // One absolute diff of 2 → mae = 2/6.
        assert!((f64::from(m.mae) - 2.0 / 6.0).abs() < 1e-6);
        assert!((f64::from(m.max_abs) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn relative_error_is_finite_for_black_reference() {
        // A nonzero candidate over a pure-black reference must not divide by
        // zero or produce a non-finite relative error.
        let reference = pixels(&[[0.0, 0.0, 0.0]]);
        let candidate = pixels(&[[1.0, 1.0, 1.0]]);
        let m = compare_pixels(&reference, &candidate).expect("equal lengths");
        assert!(m.relative_mse.is_finite());
        assert!(m.relative_mse > 0.0);
    }

    #[test]
    fn relative_error_shrinks_as_reference_brightens() {
        // The same absolute error is relatively smaller against a brighter
        // reference, so the luminance-normalized metric must decrease.
        let candidate = pixels(&[[1.1, 1.1, 1.1]]);
        let dim = pixels(&[[1.0, 1.0, 1.0]]);
        let bright = pixels(&[[10.0, 10.0, 10.0]]);
        let near = compare_pixels(&dim, &candidate).expect("equal lengths");
        let far = compare_pixels(&bright, &pixels(&[[10.1, 10.1, 10.1]])).expect("equal lengths");
        assert!(far.relative_mse < near.relative_mse);
    }

    #[test]
    fn film_dimension_mismatch_returns_none() {
        let a = Film::new(2, 2);
        let b = Film::new(3, 2);
        assert!(compare_films(&a, &b).is_none());
    }

    #[test]
    fn identical_films_are_zero() {
        let a = Film::new(2, 3);
        let b = Film::new(2, 3);
        let m = compare_films(&a, &b).expect("same dimensions");
        assert_eq!(m, ErrorMetrics::ZERO);
    }
}
