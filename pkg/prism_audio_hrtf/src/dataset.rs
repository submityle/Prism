//! In-memory model of a head-related transfer function (HRTF) dataset.
//!
//! An [`HrtfDataset`] is a set of measured head-related impulse responses
//! (HRIRs): for a grid of source directions (and optionally distances) around
//! the listener, it stores the finite impulse response from the source to the
//! left and right ear drums. Downstream code selects and interpolates these
//! responses ([`crate::interpolation`]) and convolves them with a dry signal
//! to produce a binaural (headphone) rendering ([`crate::binaural`]).
//!
//! # Layout
//!
//! Samples are stored in two flat, contiguous buffers (one per ear) so that a
//! given measurement's HRIR is a single `[Sample]` slice with no per-lookup
//! allocation. Measurement `m`'s left HRIR occupies
//! `left[m * hrir_len .. (m + 1) * hrir_len]`, and likewise for `right`. This
//! packing keeps the real-time selection/convolution path cache friendly and
//! allocation free.
//!
//! # Coordinate convention
//!
//! Directions use the same convention as [`prism_audio_spatial`]: azimuth is
//! `atan2(x, -z)` (front `= 0`, right positive, range `(-pi, pi]`) and
//! elevation is `atan2(y, hypot(x, z))` (range `[-pi/2, pi/2]`), in the
//! listener-local right-handed frame (`+X` right, `+Y` up, `-Z` forward).
//!
//! # Real-time contract
//!
//! Construction and validation ([`HrtfDataset::from_samples`]) may allocate
//! and run off the audio thread. Every accessor used on the hot path
//! ([`HrtfDataset::left_hrir`], [`HrtfDataset::right_hrir`],
//! [`HrtfDataset::measurement`]) is **allocation free, lock free, and panic
//! free** (they return `Option`/empty slices instead of panicking on a bad
//! index).
//!
//! # Determinism
//!
//! The dataset stores raw `f32` samples and angles; no transcendental math is
//! performed here, so it is trivially bit-reproducible.
//!
//! # Provenance
//!
//! This module is engine-agnostic and contains **no Unreal Engine, Unity,
//! Godot, Wwise, FMOD, or Steam Audio source or derived code**. The data model
//! is implemented from the publicly documented SOFA/AES69 conventions
//! (measurement grids of source positions with per-ear finite impulse
//! responses) using only standard collections.

use alloc::vec::Vec;
use prism_audio_core::math::Sample;

/// A single measured source position in the listener-local frame.
///
/// Angles are in radians and follow the [`prism_audio_spatial`] convention
/// (see the module docs). `distance` is the measurement radius in metres
/// (many datasets are measured on a single sphere, e.g. `1.0`).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Measurement {
    /// Azimuth in radians: `atan2(x, -z)`, front `= 0`, right positive.
    pub azimuth: Sample,
    /// Elevation in radians: `atan2(y, hypot(x, z))`, range `[-pi/2, pi/2]`.
    pub elevation: Sample,
    /// Measurement radius in metres.
    pub distance: Sample,
}

impl Measurement {
    /// Creates a measurement at the given `azimuth`/`elevation` (radians) and
    /// `distance` (metres).
    #[must_use]
    #[inline]
    pub const fn new(azimuth: Sample, elevation: Sample, distance: Sample) -> Self {
        Self {
            azimuth,
            elevation,
            distance,
        }
    }
}

/// Errors returned when constructing an [`HrtfDataset`] from raw samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatasetError {
    /// `hrir_len` was zero; an HRIR must have at least one tap.
    EmptyHrir,
    /// No measurements were supplied.
    NoMeasurements,
    /// The left sample buffer length did not equal
    /// `measurements.len() * hrir_len`.
    LeftLengthMismatch {
        /// The length that was expected.
        expected: usize,
        /// The length that was supplied.
        actual: usize,
    },
    /// The right sample buffer length did not equal
    /// `measurements.len() * hrir_len`.
    RightLengthMismatch {
        /// The length that was expected.
        expected: usize,
        /// The length that was supplied.
        actual: usize,
    },
    /// The sample rate was zero.
    ZeroSampleRate,
}

impl core::fmt::Display for DatasetError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyHrir => f.write_str("HRIR length must be non-zero"),
            Self::NoMeasurements => f.write_str("dataset must contain at least one measurement"),
            Self::LeftLengthMismatch { expected, actual } => write!(
                f,
                "left buffer length {actual} does not equal measurements * hrir_len ({expected})"
            ),
            Self::RightLengthMismatch { expected, actual } => write!(
                f,
                "right buffer length {actual} does not equal measurements * hrir_len ({expected})"
            ),
            Self::ZeroSampleRate => f.write_str("sample rate must be non-zero"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for DatasetError {}

/// An in-memory HRTF dataset: a measurement grid plus per-ear HRIRs.
///
/// See the [module documentation](self) for the storage layout and coordinate
/// convention.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct HrtfDataset {
    sample_rate: u32,
    hrir_len: usize,
    measurements: Vec<Measurement>,
    left: Vec<Sample>,
    right: Vec<Sample>,
}

impl HrtfDataset {
    /// Builds a dataset from a measurement grid and flat per-ear sample
    /// buffers.
    ///
    /// `left` and `right` must each contain exactly
    /// `measurements.len() * hrir_len` samples, laid out measurement-major
    /// (measurement `m`'s HRIR is the `hrir_len`-length slice starting at
    /// `m * hrir_len`).
    ///
    /// # Errors
    ///
    /// Returns a [`DatasetError`] if the sample rate or HRIR length is zero,
    /// if there are no measurements, or if either buffer's length does not
    /// match the grid.
    pub fn from_samples(
        sample_rate: u32,
        hrir_len: usize,
        measurements: Vec<Measurement>,
        left: Vec<Sample>,
        right: Vec<Sample>,
    ) -> Result<Self, DatasetError> {
        if sample_rate == 0 {
            return Err(DatasetError::ZeroSampleRate);
        }
        if hrir_len == 0 {
            return Err(DatasetError::EmptyHrir);
        }
        if measurements.is_empty() {
            return Err(DatasetError::NoMeasurements);
        }
        let expected = measurements.len() * hrir_len;
        if left.len() != expected {
            return Err(DatasetError::LeftLengthMismatch {
                expected,
                actual: left.len(),
            });
        }
        if right.len() != expected {
            return Err(DatasetError::RightLengthMismatch {
                expected,
                actual: right.len(),
            });
        }
        Ok(Self {
            sample_rate,
            hrir_len,
            measurements,
            left,
            right,
        })
    }

    /// The sample rate of the stored HRIRs, in Hz.
    #[must_use]
    #[inline]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// The number of taps in every HRIR.
    #[must_use]
    #[inline]
    pub fn hrir_len(&self) -> usize {
        self.hrir_len
    }

    /// The number of measurement points in the grid.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.measurements.len()
    }

    /// Returns `true` if the dataset has no measurements.
    ///
    /// Always `false` for a dataset built by [`HrtfDataset::from_samples`]
    /// (which rejects an empty grid); provided for completeness.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.measurements.is_empty()
    }

    /// The full measurement grid.
    #[must_use]
    #[inline]
    pub fn measurements(&self) -> &[Measurement] {
        &self.measurements
    }

    /// The measurement at `index`, or `None` if out of range.
    #[must_use]
    #[inline]
    pub fn measurement(&self, index: usize) -> Option<Measurement> {
        self.measurements.get(index).copied()
    }

    /// The left-ear HRIR for measurement `index`, or an empty slice if the
    /// index is out of range.
    ///
    /// Real-time safe: no allocation, no panic.
    #[must_use]
    #[inline]
    pub fn left_hrir(&self, index: usize) -> &[Sample] {
        self.hrir_slice(&self.left, index)
    }

    /// The right-ear HRIR for measurement `index`, or an empty slice if the
    /// index is out of range.
    ///
    /// Real-time safe: no allocation, no panic.
    #[must_use]
    #[inline]
    pub fn right_hrir(&self, index: usize) -> &[Sample] {
        self.hrir_slice(&self.right, index)
    }

    #[inline]
    fn hrir_slice<'a>(&self, buf: &'a [Sample], index: usize) -> &'a [Sample] {
        let start = index * self.hrir_len;
        let end = start + self.hrir_len;
        buf.get(start..end).unwrap_or(&[])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn tiny_dataset() -> HrtfDataset {
        // Two measurements, 3-tap HRIRs.
        let measurements = vec![
            Measurement::new(0.0, 0.0, 1.0),
            Measurement::new(1.0, 0.0, 1.0),
        ];
        let left = vec![1.0, 0.0, 0.0, 0.5, 0.5, 0.0];
        let right = vec![0.0, 1.0, 0.0, 0.0, 0.5, 0.5];
        HrtfDataset::from_samples(48_000, 3, measurements, left, right).unwrap()
    }

    #[test]
    fn builds_and_reports_geometry() {
        let ds = tiny_dataset();
        assert_eq!(ds.sample_rate(), 48_000);
        assert_eq!(ds.hrir_len(), 3);
        assert_eq!(ds.len(), 2);
        assert!(!ds.is_empty());
    }

    #[test]
    fn slices_are_measurement_major() {
        let ds = tiny_dataset();
        assert_eq!(ds.left_hrir(0), &[1.0, 0.0, 0.0]);
        assert_eq!(ds.right_hrir(0), &[0.0, 1.0, 0.0]);
        assert_eq!(ds.left_hrir(1), &[0.5, 0.5, 0.0]);
        assert_eq!(ds.right_hrir(1), &[0.0, 0.5, 0.5]);
    }

    #[test]
    fn out_of_range_index_is_empty_not_panic() {
        let ds = tiny_dataset();
        assert!(ds.left_hrir(99).is_empty());
        assert!(ds.right_hrir(99).is_empty());
        assert_eq!(ds.measurement(99), None);
    }

    #[test]
    fn measurement_lookup_round_trips() {
        let ds = tiny_dataset();
        assert_eq!(ds.measurement(0), Some(Measurement::new(0.0, 0.0, 1.0)));
        assert_eq!(ds.measurement(1), Some(Measurement::new(1.0, 0.0, 1.0)));
    }

    #[test]
    fn rejects_zero_sample_rate() {
        let err = HrtfDataset::from_samples(
            0,
            1,
            vec![Measurement::new(0.0, 0.0, 1.0)],
            vec![0.0],
            vec![0.0],
        );
        assert_eq!(err.unwrap_err(), DatasetError::ZeroSampleRate);
    }

    #[test]
    fn rejects_empty_hrir() {
        let err = HrtfDataset::from_samples(
            48_000,
            0,
            vec![Measurement::new(0.0, 0.0, 1.0)],
            vec![],
            vec![],
        );
        assert_eq!(err.unwrap_err(), DatasetError::EmptyHrir);
    }

    #[test]
    fn rejects_no_measurements() {
        let err = HrtfDataset::from_samples(48_000, 2, vec![], vec![], vec![]);
        assert_eq!(err.unwrap_err(), DatasetError::NoMeasurements);
    }

    #[test]
    fn rejects_length_mismatch() {
        let err = HrtfDataset::from_samples(
            48_000,
            3,
            vec![Measurement::new(0.0, 0.0, 1.0)],
            vec![1.0, 0.0],
            vec![0.0, 0.0, 0.0],
        );
        assert_eq!(
            err.unwrap_err(),
            DatasetError::LeftLengthMismatch {
                expected: 3,
                actual: 2
            }
        );
    }

    #[test]
    fn error_display_is_non_empty() {
        let err = DatasetError::EmptyHrir;
        let mut s = String::new();
        use core::fmt::Write;
        write!(s, "{err}").unwrap();
        assert!(!s.is_empty());
    }
}
