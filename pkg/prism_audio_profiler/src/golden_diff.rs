//! Sample-accurate comparison of a candidate render against a golden reference.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the golden-diff part of design section 26, the regression check
//! used by the deterministic-render test harness (design section 24). It
//! compares two sample buffers and reports the maximum absolute error, the
//! RMS error, the index of first divergence, and the maximum unit-in-the-last-
//! place (ULP) distance. The ULP metric uses the safe `f32::to_bits` total
//! ordering so no unsafe code is required.

use prism_audio_core::math::Sample;

use bevy_math::ops;

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

/// Tolerances that classify a [`GoldenDiff`] as passing.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct GoldenDiffConfig {
    /// Maximum permitted absolute per-sample error.
    pub abs_tolerance: Sample,
    /// Maximum permitted root-mean-square error across the buffer.
    pub rms_tolerance: Sample,
    /// Maximum permitted ULP distance for any single sample.
    pub ulp_tolerance: u64,
}

impl GoldenDiffConfig {
    /// A strict configuration that permits only bit-identical buffers.
    #[must_use]
    #[inline]
    pub const fn exact() -> Self {
        Self {
            abs_tolerance: 0.0,
            rms_tolerance: 0.0,
            ulp_tolerance: 0,
        }
    }

    /// A configuration with the given absolute and RMS tolerances and an
    /// unbounded ULP budget.
    #[must_use]
    #[inline]
    pub const fn tolerant(abs_tolerance: Sample, rms_tolerance: Sample) -> Self {
        Self {
            abs_tolerance,
            rms_tolerance,
            ulp_tolerance: u64::MAX,
        }
    }
}

impl Default for GoldenDiffConfig {
    #[inline]
    fn default() -> Self {
        Self::exact()
    }
}

/// The result of comparing a candidate buffer against a reference buffer.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct GoldenDiff {
    /// Number of sample positions compared (the shorter of the two lengths).
    pub compared: usize,
    /// `true` when the two buffers had different lengths.
    pub length_mismatch: bool,
    /// Maximum absolute per-sample error.
    pub max_abs_error: Sample,
    /// Root-mean-square error across the compared samples.
    pub rms_error: Sample,
    /// Index of the first diverging sample, if any.
    pub first_divergence: Option<usize>,
    /// Maximum ULP distance observed for any compared sample.
    pub max_ulp: u64,
}

/// Map a finite or non-finite `f32` to a monotonic integer key so that ULP
/// distance can be computed as an integer difference across the real line.
fn ordered_key(value: f32) -> i64 {
    let bits = value.to_bits();
    if bits & 0x8000_0000 == 0 {
        i64::from(bits)
    } else {
        0x8000_0000_i64 - i64::from(bits)
    }
}

/// ULP distance between two `f32` values using the total-ordering key.
fn ulp_distance(a: f32, b: f32) -> u64 {
    let ka = ordered_key(a);
    let kb = ordered_key(b);
    (ka - kb).unsigned_abs()
}

impl GoldenDiff {
    /// Compare `candidate` against `reference`, computing all error metrics.
    ///
    /// Comparison runs over the common prefix of the two slices; a length
    /// difference sets [`GoldenDiff::length_mismatch`]. Divergence is any
    /// sample whose absolute error exceeds the configured absolute tolerance.
    #[must_use]
    pub fn compare(reference: &[Sample], candidate: &[Sample], config: GoldenDiffConfig) -> Self {
        let compared = reference.len().min(candidate.len());
        let length_mismatch = reference.len() != candidate.len();

        let mut max_abs_error: Sample = 0.0;
        let mut sum_sq: f64 = 0.0;
        let mut first_divergence: Option<usize> = None;
        let mut max_ulp: u64 = 0;

        for index in 0..compared {
            let r = reference[index];
            let c = candidate[index];
            let error = (c - r).abs();
            if error > max_abs_error {
                max_abs_error = error;
            }
            sum_sq += f64::from(error) * f64::from(error);
            let ulp = ulp_distance(r, c);
            if ulp > max_ulp {
                max_ulp = ulp;
            }
            if first_divergence.is_none() && error > config.abs_tolerance {
                first_divergence = Some(index);
            }
        }

        let rms_error = if compared == 0 {
            0.0
        } else {
            ops::sqrt((sum_sq / compared as f64) as f32)
        };

        Self {
            compared,
            length_mismatch,
            max_abs_error,
            rms_error,
            first_divergence,
            max_ulp,
        }
    }

    /// `true` when the diff is within the given tolerances and the buffers had
    /// equal length.
    #[must_use]
    pub fn within(&self, config: GoldenDiffConfig) -> bool {
        !self.length_mismatch
            && self.max_abs_error <= config.abs_tolerance
            && self.rms_error <= config.rms_tolerance
            && self.max_ulp <= config.ulp_tolerance
    }

    /// `true` when no sample diverged beyond the absolute tolerance used to
    /// build this diff and the buffers had equal length.
    #[must_use]
    #[inline]
    pub fn is_match(&self) -> bool {
        !self.length_mismatch && self.first_divergence.is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: Sample = 1e-6;

    #[test]
    fn identical_buffers_match_exactly() {
        let reference = [0.0, 0.25, -0.5, 1.0];
        let diff = GoldenDiff::compare(&reference, &reference, GoldenDiffConfig::exact());
        assert_eq!(diff.compared, 4);
        assert!(!diff.length_mismatch);
        assert!(diff.max_abs_error.abs() < EPS);
        assert!(diff.rms_error.abs() < EPS);
        assert_eq!(diff.first_divergence, None);
        assert_eq!(diff.max_ulp, 0);
        assert!(diff.is_match());
        assert!(diff.within(GoldenDiffConfig::exact()));
    }

    #[test]
    fn divergence_index_is_first_offender() {
        let reference = [0.0, 0.5, 0.5, 0.5];
        let candidate = [0.0, 0.5, 0.6, 0.5];
        let diff = GoldenDiff::compare(&reference, &candidate, GoldenDiffConfig::exact());
        assert_eq!(diff.first_divergence, Some(2));
        assert!((diff.max_abs_error - 0.1).abs() < 1e-4);
        assert!(!diff.is_match());
    }

    #[test]
    fn rms_error_is_averaged() {
        let reference = [0.0, 0.0, 0.0, 0.0];
        let candidate = [0.2, 0.0, 0.0, 0.0];
        let diff = GoldenDiff::compare(&reference, &candidate, GoldenDiffConfig::exact());
        // RMS of [0.2,0,0,0] = sqrt(0.04/4) = 0.1.
        assert!((diff.rms_error - 0.1).abs() < 1e-4);
    }

    #[test]
    fn tolerant_config_accepts_small_error() {
        let reference = [0.0, 0.5, -0.5];
        let candidate = [0.0005, 0.4998, -0.5003];
        let config = GoldenDiffConfig::tolerant(1e-3, 1e-3);
        let diff = GoldenDiff::compare(&reference, &candidate, config);
        assert!(diff.within(config));
    }

    #[test]
    fn length_mismatch_flagged_but_prefix_compared() {
        let reference = [0.0, 0.5];
        let candidate = [0.0, 0.5, 0.9];
        let diff = GoldenDiff::compare(&reference, &candidate, GoldenDiffConfig::exact());
        assert_eq!(diff.compared, 2);
        assert!(diff.length_mismatch);
        assert!(!diff.within(GoldenDiffConfig::exact()));
    }

    #[test]
    fn empty_buffers_produce_zero_metrics() {
        let diff = GoldenDiff::compare(&[], &[], GoldenDiffConfig::exact());
        assert_eq!(diff.compared, 0);
        assert!(!diff.length_mismatch);
        assert_eq!(diff.max_ulp, 0);
        assert!(diff.is_match());
    }

    #[test]
    fn ulp_distance_of_adjacent_floats_is_one() {
        let a: f32 = 1.0;
        let b = f32::from_bits(a.to_bits() + 1);
        let diff = GoldenDiff::compare(&[a], &[b], GoldenDiffConfig::tolerant(1.0, 1.0));
        assert_eq!(diff.max_ulp, 1);
    }

    #[test]
    fn ulp_distance_crosses_zero_symmetrically() {
        // Smallest positive and smallest negative subnormal are two ULPs apart
        // across signed zero.
        let pos = f32::from_bits(1);
        let neg = f32::from_bits(0x8000_0001);
        let diff = GoldenDiff::compare(&[pos], &[neg], GoldenDiffConfig::tolerant(1.0, 1.0));
        assert_eq!(diff.max_ulp, 2);
    }
}
