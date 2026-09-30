//! Loading HRTF datasets from external sources, including the SOFA/AES69
//! interchange format.
//!
//! The heavy lifting of turning measured data into a renderable
//! [`HrtfDataset`] is split into two concerns:
//!
//! 1. A [`HrirSource`] trait: any producer of per-measurement HRIR records
//!    (a decoded SOFA file, a procedural synthesizer, a unit-test fixture).
//! 2. [`build_dataset`], which assembles records into the flat, validated
//!    [`HrtfDataset`] layout the renderer consumes.
//!
//! # SOFA / AES69 status
//!
//! SOFA ("Spatially Oriented Format for Acoustics", standardized as
//! **AES69-2022**) stores HRIRs inside a **netCDF-4 / HDF5** binary container
//! with conventions such as `SimpleFreeFieldHRIR`. A full, dependency-free
//! HDF5 reader is a large undertaking; rather than ship a fragile partial
//! parser, this module provides:
//!
//! - [`SofaConvention`] / [`SofaRecord`]: the *decoded* logical model
//!   (source positions in the AES69 spherical system plus per-ear FIR data),
//!   independent of the binary container.
//! - [`SofaRecords`]: a [`HrirSource`] over a collection of decoded records,
//!   converting the AES69 spherical convention into this crate's
//!   listener-local azimuth/elevation convention.
//! - [`build_dataset`]: the shared assembly + validation path.
//!
//! Binary HDF5/netCDF-4 decoding into [`SofaRecord`]s is intentionally **left
//! as a follow-up** (it belongs behind the `std` feature and an optional HDF5
//! dependency). Everything downstream of the decoded records - conversion,
//! assembly, interpolation, convolution - is fully implemented and testable
//! here. This is an honest boundary, not a stub of the DSP core.
//!
//! # Real-time contract
//!
//! Loading/assembly allocates and must run off the audio thread. Nothing here
//! is called from the render callback.
//!
//! # Provenance
//!
//! This module is engine-agnostic and contains **no Unreal Engine, Unity,
//! Godot, Wwise, FMOD, or Steam Audio source or derived code**. The AES69
//! spherical-to-local angle conversion and dataset assembly are implemented
//! from the publicly documented SOFA/AES69 specification using only standard
//! collections and [`bevy_math::ops`].

use alloc::vec::Vec;
use bevy_math::ops;
use core::f32::consts::{FRAC_PI_2, PI};
use prism_audio_core::math::Sample;

use crate::dataset::{DatasetError, HrtfDataset, Measurement};

/// One measured HRIR record, in this crate's listener-local convention.
///
/// Yielded by a [`HrirSource`]. `left` and `right` must have equal length; the
/// common length becomes the dataset's `hrir_len`.
#[derive(Debug, Clone, PartialEq)]
pub struct HrirRecord {
    /// Source direction/distance for this record.
    pub measurement: Measurement,
    /// Left-ear finite impulse response.
    pub left: Vec<Sample>,
    /// Right-ear finite impulse response.
    pub right: Vec<Sample>,
}

/// A producer of HRIR records plus the sample rate they were measured at.
///
/// Implementors decode some external representation (a SOFA file, a synthetic
/// model, a fixture) into logical records; [`build_dataset`] then validates
/// and packs them.
pub trait HrirSource {
    /// The sample rate of the produced HRIRs, in Hz.
    fn sample_rate(&self) -> u32;

    /// Consumes the source into its records.
    fn into_records(self) -> Vec<HrirRecord>;
}

/// Errors that can occur while assembling a dataset from a [`HrirSource`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadError {
    /// The source produced no records.
    NoRecords,
    /// A record's left/right FIR lengths differed, or differed from the first
    /// record's length.
    InconsistentHrirLength {
        /// The length established by the first record.
        expected: usize,
        /// The offending record's index.
        record: usize,
    },
    /// Assembly produced an invalid [`HrtfDataset`].
    Dataset(DatasetError),
}

impl core::fmt::Display for LoadError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoRecords => f.write_str("HRIR source produced no records"),
            Self::InconsistentHrirLength { expected, record } => write!(
                f,
                "record {record} has an HRIR length differing from the expected {expected}"
            ),
            Self::Dataset(e) => write!(f, "dataset assembly failed: {e}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for LoadError {}

/// Assembles the records from a [`HrirSource`] into a validated
/// [`HrtfDataset`].
///
/// All records must share a single HRIR length (SOFA datasets are stored on a
/// fixed tap count); the left and right FIRs of each record must match that
/// length.
///
/// # Errors
///
/// Returns a [`LoadError`] if the source is empty, if any record has an
/// inconsistent FIR length, or if the resulting dataset fails validation.
pub fn build_dataset<S: HrirSource>(source: S) -> Result<HrtfDataset, LoadError> {
    let sample_rate = source.sample_rate();
    let records = source.into_records();
    if records.is_empty() {
        return Err(LoadError::NoRecords);
    }

    let hrir_len = records[0].left.len();
    let mut measurements = Vec::with_capacity(records.len());
    let mut left = Vec::with_capacity(records.len() * hrir_len);
    let mut right = Vec::with_capacity(records.len() * hrir_len);

    for (i, record) in records.into_iter().enumerate() {
        if record.left.len() != hrir_len || record.right.len() != hrir_len {
            return Err(LoadError::InconsistentHrirLength { expected: hrir_len, record: i });
        }
        measurements.push(record.measurement);
        left.extend_from_slice(&record.left);
        right.extend_from_slice(&record.right);
    }

    HrtfDataset::from_samples(sample_rate, hrir_len, measurements, left, right)
        .map_err(LoadError::Dataset)
}

/// The SOFA convention a set of decoded records was measured under.
///
/// Only the free-field HRIR conventions are modelled; they share the same
/// spherical `SourcePosition` system (azimuth counter-clockwise from front in
/// the horizontal plane, elevation up, radius in metres).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum SofaConvention {
    /// `SimpleFreeFieldHRIR`: a single listener, free-field HRIRs on a
    /// spherical source grid.
    SimpleFreeFieldHrir,
}

/// A decoded SOFA measurement in the **AES69 spherical** source-position
/// system, prior to conversion into this crate's local convention.
///
/// In AES69 spherical coordinates, `azimuth_deg` runs counter-clockwise from
/// the front (`0` = front, `90` = left) and `elevation_deg` is positive up.
#[derive(Debug, Clone, PartialEq)]
pub struct SofaRecord {
    /// AES69 azimuth in degrees, counter-clockwise from front (`0..360`).
    pub azimuth_deg: Sample,
    /// AES69 elevation in degrees, positive up (`-90..=90`).
    pub elevation_deg: Sample,
    /// Source radius in metres.
    pub radius_m: Sample,
    /// Left-ear FIR.
    pub left: Vec<Sample>,
    /// Right-ear FIR.
    pub right: Vec<Sample>,
}

/// A [`HrirSource`] over decoded SOFA records.
///
/// Construct this once a SOFA file (or other AES69-conformant source) has been
/// decoded into [`SofaRecord`]s; it performs the spherical-to-local angle
/// conversion when producing records.
#[derive(Debug, Clone, PartialEq)]
pub struct SofaRecords {
    sample_rate: u32,
    convention: SofaConvention,
    records: Vec<SofaRecord>,
}

impl SofaRecords {
    /// Creates a source from decoded records measured at `sample_rate` under
    /// `convention`.
    #[must_use]
    #[inline]
    pub fn new(sample_rate: u32, convention: SofaConvention, records: Vec<SofaRecord>) -> Self {
        Self { sample_rate, convention, records }
    }

    /// The convention these records were measured under.
    #[must_use]
    #[inline]
    pub fn convention(&self) -> SofaConvention {
        self.convention
    }
}

/// Converts an AES69 spherical azimuth/elevation (degrees) into this crate's
/// listener-local convention (radians).
///
/// AES69 azimuth is counter-clockwise from front (`+90` = left), whereas the
/// local convention (`atan2(x, -z)`) has right positive, so the azimuth sign
/// is inverted and the result is wrapped to `(-pi, pi]`. Elevation keeps its
/// sign (positive up) and is only converted to radians.
#[must_use]
pub fn aes69_to_local(azimuth_deg: Sample, elevation_deg: Sample) -> (Sample, Sample) {
    const DEG_TO_RAD: Sample = PI / 180.0;
    // AES69 left-positive -> local right-positive.
    let mut az = -azimuth_deg * DEG_TO_RAD;
    // Wrap into (-pi, pi].
    let two_pi = 2.0 * PI;
    az = az - two_pi * ops::floor((az + PI) / two_pi);
    if az <= -PI {
        az += two_pi;
    }
    let el = (elevation_deg * DEG_TO_RAD).clamp(-FRAC_PI_2, FRAC_PI_2);
    (az, el)
}

impl HrirSource for SofaRecords {
    #[inline]
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn into_records(self) -> Vec<HrirRecord> {
        let mut out = Vec::with_capacity(self.records.len());
        for r in self.records {
            let (azimuth, elevation) = aes69_to_local(r.azimuth_deg, r.elevation_deg);
            out.push(HrirRecord {
                measurement: Measurement::new(azimuth, elevation, r.radius_m),
                left: r.left,
                right: r.right,
            });
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use core::f32::consts::FRAC_PI_2;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    struct FixtureSource {
        records: Vec<HrirRecord>,
    }

    impl HrirSource for FixtureSource {
        fn sample_rate(&self) -> u32 {
            48_000
        }
        fn into_records(self) -> Vec<HrirRecord> {
            self.records
        }
    }

    #[test]
    fn builds_dataset_from_records() {
        let source = FixtureSource {
            records: vec![
                HrirRecord {
                    measurement: Measurement::new(0.0, 0.0, 1.0),
                    left: vec![1.0, 0.0],
                    right: vec![0.0, 1.0],
                },
                HrirRecord {
                    measurement: Measurement::new(0.5, 0.0, 1.0),
                    left: vec![0.5, 0.5],
                    right: vec![0.5, 0.5],
                },
            ],
        };
        let ds = build_dataset(source).unwrap();
        assert_eq!(ds.len(), 2);
        assert_eq!(ds.hrir_len(), 2);
        assert_eq!(ds.left_hrir(0), &[1.0, 0.0]);
    }

    #[test]
    fn rejects_empty_source() {
        let source = FixtureSource { records: vec![] };
        assert_eq!(build_dataset(source).unwrap_err(), LoadError::NoRecords);
    }

    #[test]
    fn rejects_inconsistent_lengths() {
        let source = FixtureSource {
            records: vec![
                HrirRecord {
                    measurement: Measurement::new(0.0, 0.0, 1.0),
                    left: vec![1.0, 0.0],
                    right: vec![0.0, 1.0],
                },
                HrirRecord {
                    measurement: Measurement::new(0.5, 0.0, 1.0),
                    left: vec![0.5, 0.5, 0.0],
                    right: vec![0.5, 0.5, 0.0],
                },
            ],
        };
        assert_eq!(
            build_dataset(source).unwrap_err(),
            LoadError::InconsistentHrirLength { expected: 2, record: 1 }
        );
    }

    #[test]
    fn rejects_mismatched_left_right() {
        let source = FixtureSource {
            records: vec![HrirRecord {
                measurement: Measurement::new(0.0, 0.0, 1.0),
                left: vec![1.0, 0.0],
                right: vec![0.0],
            }],
        };
        assert_eq!(
            build_dataset(source).unwrap_err(),
            LoadError::InconsistentHrirLength { expected: 2, record: 0 }
        );
    }

    #[test]
    fn aes69_front_maps_to_zero() {
        let (az, el) = aes69_to_local(0.0, 0.0);
        assert!(approx(az, 0.0, 1e-6));
        assert!(approx(el, 0.0, 1e-6));
    }

    #[test]
    fn aes69_left_is_negative_local_azimuth() {
        // AES69 +90 deg = left = local -pi/2 (right positive).
        let (az, _) = aes69_to_local(90.0, 0.0);
        assert!(approx(az, -FRAC_PI_2, 1e-5));
    }

    #[test]
    fn aes69_right_is_positive_local_azimuth() {
        // AES69 270 deg (== -90) = right = local +pi/2.
        let (az, _) = aes69_to_local(270.0, 0.0);
        assert!(approx(az, FRAC_PI_2, 1e-5));
    }

    #[test]
    fn aes69_elevation_is_clamped_and_converted() {
        let (_, el) = aes69_to_local(0.0, 90.0);
        assert!(approx(el, FRAC_PI_2, 1e-6));
        let (_, el_over) = aes69_to_local(0.0, 200.0);
        assert!(approx(el_over, FRAC_PI_2, 1e-6));
    }

    #[test]
    fn sofa_records_source_converts_and_builds() {
        let records = vec![
            SofaRecord {
                azimuth_deg: 0.0,
                elevation_deg: 0.0,
                radius_m: 1.0,
                left: vec![1.0, 0.0],
                right: vec![1.0, 0.0],
            },
            SofaRecord {
                azimuth_deg: 90.0,
                elevation_deg: 0.0,
                radius_m: 1.0,
                left: vec![0.0, 1.0],
                right: vec![0.0, 1.0],
            },
        ];
        let source = SofaRecords::new(44_100, SofaConvention::SimpleFreeFieldHrir, records);
        assert_eq!(source.convention(), SofaConvention::SimpleFreeFieldHrir);
        let ds = build_dataset(source).unwrap();
        assert_eq!(ds.sample_rate(), 44_100);
        assert_eq!(ds.len(), 2);
        // Second record was AES69 left (+90) -> local -pi/2.
        assert!(approx(ds.measurement(1).unwrap().azimuth, -FRAC_PI_2, 1e-5));
    }
}
