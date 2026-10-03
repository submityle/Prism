//! Turning a decoded HDF5 container ([`crate::sofa::hdf5::H5File`]) into the
//! logical SOFA model this crate renders from ([`SofaRecords`]).
//!
//! This is the bridge between the dependency-free HDF5 subset codec and the
//! HRIR pipeline: it validates that a file is a `SimpleFreeFieldHRIR` dataset,
//! reads the three datasets a free-field HRIR file carries (`Data.IR`,
//! `Data.SamplingRate`, `SourcePosition`), and emits one [`SofaRecord`] per
//! measurement in the AES69 spherical system. Angle conversion into this
//! crate's listener-local convention is deferred to [`SofaRecords`] (which
//! calls [`crate::sofa::aes69_to_local`]), so the decoder stays a pure,
//! coordinate-preserving reshape.
//!
//! # Supported subset
//!
//! - Convention `SimpleFreeFieldHRIR` (checked via the `SOFAConventions`
//!   attribute when present; otherwise inferred from the dataset shapes).
//! - `Data.IR` with shape `[M][R][N]`, `R == 2` (receiver 0 = left ear,
//!   receiver 1 = right ear), `N` taps.
//! - `SourcePosition` with shape `[M][3]` or `[1][3]` (shared position),
//!   columns `(azimuth_deg, elevation_deg, radius_m)` in the AES69 spherical
//!   system.
//! - `Data.SamplingRate` as a scalar / single-element dataset in Hz.
//!
//! Anything outside this subset is reported as a typed [`DecodeError`]; the
//! decoder never fabricates or silently drops measurements.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, or Steam Audio
//! source or derived code; no AI/ML. The mapping follows the publicly
//! published AES69 `SimpleFreeFieldHRIR` convention.
//!
//! # Relationship
//!
//! Consumes [`crate::sofa::hdf5::H5File`] (the raw arrays) and produces a
//! [`SofaRecords`] [`crate::sofa::HrirSource`], which [`crate::sofa::build_dataset`]
//! then assembles into an [`crate::dataset::HrtfDataset`]. Entirely `std`-feature
//! gated and off the real-time path.

use alloc::string::String;
use alloc::vec::Vec;
use prism_audio_core::math::Sample;

use crate::sofa::hdf5::{H5Error, H5File};
use crate::sofa::{SofaConvention, SofaRecord, SofaRecords};

/// The dataset name holding the HRIR tensor, shape `[M][R][N]`.
const DATASET_IR: &str = "Data.IR";
/// The dataset name holding the per-measurement source positions.
const DATASET_SOURCE_POSITION: &str = "SourcePosition";
/// The dataset name holding the sampling rate in Hz.
const DATASET_SAMPLING_RATE: &str = "Data.SamplingRate";
/// The root attribute naming the SOFA convention.
const ATTR_SOFA_CONVENTIONS: &str = "SOFAConventions";
/// The expected `SOFAConventions` value for free-field HRIRs.
const CONVENTION_SIMPLE_FREE_FIELD_HRIR: &str = "SimpleFreeFieldHRIR";
/// The number of receivers (ears) a `SimpleFreeFieldHRIR` file carries.
const RECEIVER_COUNT: u64 = 2;
/// The number of coordinate columns in a spherical `SourcePosition`.
const POSITION_COLUMNS: u64 = 3;

/// Errors produced while decoding an [`H5File`] into [`SofaRecords`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    /// The underlying HDF5 container could not be read.
    Container(H5Error),
    /// The file declares a `SOFAConventions` this decoder does not model.
    UnsupportedConvention(String),
    /// A required dataset was absent from the container.
    MissingDataset(&'static str),
    /// A dataset had a rank or extent the convention does not permit.
    BadShape {
        /// The dataset whose shape was rejected.
        dataset: &'static str,
    },
    /// The measurement count implied by two datasets disagreed.
    InconsistentMeasurementCount {
        /// Measurement count implied by `Data.IR`.
        ir: u64,
        /// Measurement count implied by `SourcePosition`.
        source_position: u64,
    },
    /// The sampling-rate dataset was empty or non-positive.
    InvalidSamplingRate,
}

impl core::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Container(e) => write!(f, "HDF5 container error: {e}"),
            Self::UnsupportedConvention(name) => {
                write!(f, "unsupported SOFA convention \"{name}\"")
            }
            Self::MissingDataset(name) => write!(f, "required dataset \"{name}\" is missing"),
            Self::BadShape { dataset } => {
                write!(f, "dataset \"{dataset}\" has an unsupported shape")
            }
            Self::InconsistentMeasurementCount {
                ir,
                source_position,
            } => write!(
                f,
                "measurement count mismatch: Data.IR has {ir}, SourcePosition has {source_position}"
            ),
            Self::InvalidSamplingRate => {
                f.write_str("Data.SamplingRate is empty or non-positive")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for DecodeError {}

impl From<H5Error> for DecodeError {
    #[inline]
    fn from(e: H5Error) -> Self {
        Self::Container(e)
    }
}

/// Decodes a SOFA `SimpleFreeFieldHRIR` byte image into [`SofaRecords`].
///
/// The returned source yields one record per measurement, carrying the AES69
/// spherical angles verbatim; the spherical-to-local conversion happens when
/// the records are consumed by [`crate::sofa::build_dataset`].
///
/// # Errors
///
/// Returns a [`DecodeError`] if the container cannot be read, declares an
/// unsupported convention, is missing a required dataset, or has shapes that
/// violate the `SimpleFreeFieldHRIR` layout.
pub fn decode_sofa(bytes: &[u8]) -> Result<SofaRecords, DecodeError> {
    let file = H5File::read(bytes)?;
    decode_file(&file)
}

/// Decodes an already-read [`H5File`] into [`SofaRecords`].
///
/// Separated from [`decode_sofa`] so callers that already hold a parsed file
/// (for example after inspecting attributes) can reuse it without re-parsing.
///
/// # Errors
///
/// See [`decode_sofa`].
pub fn decode_file(file: &H5File) -> Result<SofaRecords, DecodeError> {
    check_convention(file)?;

    let ir = file
        .datasets
        .get(DATASET_IR)
        .ok_or(DecodeError::MissingDataset(DATASET_IR))?;
    let positions = file
        .datasets
        .get(DATASET_SOURCE_POSITION)
        .ok_or(DecodeError::MissingDataset(DATASET_SOURCE_POSITION))?;
    let rate = file
        .datasets
        .get(DATASET_SAMPLING_RATE)
        .ok_or(DecodeError::MissingDataset(DATASET_SAMPLING_RATE))?;

    // Data.IR must be [M][R][N] with R == 2.
    if ir.dims.len() != 3 || ir.dims[1] != RECEIVER_COUNT {
        return Err(DecodeError::BadShape {
            dataset: DATASET_IR,
        });
    }
    let measurements = ir.dims[0];
    let taps = ir.dims[2];

    // SourcePosition must be [M][3] or [1][3].
    if positions.dims.len() != 2 || positions.dims[1] != POSITION_COLUMNS {
        return Err(DecodeError::BadShape {
            dataset: DATASET_SOURCE_POSITION,
        });
    }
    let position_rows = positions.dims[0];
    let shared_position = position_rows == 1;
    if !shared_position && position_rows != measurements {
        return Err(DecodeError::InconsistentMeasurementCount {
            ir: measurements,
            source_position: position_rows,
        });
    }

    let sample_rate = read_sampling_rate(rate)?;

    let taps = usize_from(taps);
    let measurement_count = usize_from(measurements);
    let mut records = Vec::with_capacity(measurement_count);

    // Row-major strides: IR element (m, r, n) sits at ((m * R) + r) * N + n.
    let receivers = usize_from(RECEIVER_COUNT);
    let stride_measurement = receivers * taps;

    for m in 0..measurement_count {
        let base = m * stride_measurement;
        let left_start = base;
        let right_start = base + taps;

        let mut left = Vec::with_capacity(taps);
        let mut right = Vec::with_capacity(taps);
        for n in 0..taps {
            left.push(ir.data[left_start + n] as Sample);
            right.push(ir.data[right_start + n] as Sample);
        }

        let pos_row = if shared_position { 0 } else { m };
        let pos_base = pos_row * usize_from(POSITION_COLUMNS);
        let azimuth_deg = positions.data[pos_base] as Sample;
        let elevation_deg = positions.data[pos_base + 1] as Sample;
        let radius_m = positions.data[pos_base + 2] as Sample;

        records.push(SofaRecord {
            azimuth_deg,
            elevation_deg,
            radius_m,
            left,
            right,
        });
    }

    Ok(SofaRecords::new(
        sample_rate,
        SofaConvention::SimpleFreeFieldHrir,
        records,
    ))
}

/// Validates the `SOFAConventions` attribute when present.
///
/// A missing attribute is tolerated (some minimal exporters omit it); a
/// present-but-wrong value is rejected so unrelated HDF5 files are not decoded
/// as free-field HRIRs.
fn check_convention(file: &H5File) -> Result<(), DecodeError> {
    if let Some(value) = file.attributes.get(ATTR_SOFA_CONVENTIONS)
        && value != CONVENTION_SIMPLE_FREE_FIELD_HRIR
    {
        return Err(DecodeError::UnsupportedConvention(value.clone()));
    }
    Ok(())
}

/// Reads a strictly positive sampling rate from a scalar / single-element
/// dataset, rounding to the nearest hertz.
fn read_sampling_rate(dataset: &crate::sofa::hdf5::Dataset) -> Result<u32, DecodeError> {
    let value = *dataset.data.first().ok_or(DecodeError::InvalidSamplingRate)?;
    if !(value.is_finite() && value > 0.0) {
        return Err(DecodeError::InvalidSamplingRate);
    }
    // Round half-up; sampling rates are integer hertz in practice.
    let rounded = (value + 0.5) as u64;
    if rounded == 0 || rounded > u64::from(u32::MAX) {
        return Err(DecodeError::InvalidSamplingRate);
    }
    Ok(rounded as u32)
}

/// Narrows a `u64` extent to `usize` saturating at `usize::MAX`.
///
/// Extents originate from an already-validated in-memory [`H5File`] whose data
/// length equals the product of its dims, so on 64-bit targets this is lossless
/// for any realistic SOFA file; the saturation only guards 32-bit builds.
#[inline]
fn usize_from(value: u64) -> usize {
    usize::try_from(value).unwrap_or(usize::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sofa::hdf5::{Dataset, Dtype, H5File};
    use crate::sofa::{build_dataset, HrirSource};
    use alloc::vec;
    use core::f32::consts::FRAC_PI_2;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    /// Builds a two-measurement `SimpleFreeFieldHRIR` file: front and left.
    fn fixture_file(ir_chunked: bool) -> H5File {
        let mut file = H5File::new();
        file.set_attribute("Conventions", "SOFA");
        file.set_attribute(ATTR_SOFA_CONVENTIONS, CONVENTION_SIMPLE_FREE_FIELD_HRIR);

        // M=2, R=2, N=3.
        // front: left=[1,0,0], right=[0,1,0]; left: left=[0,0,1], right=[1,1,1]
        let ir_data = vec![
            1.0, 0.0, 0.0, // m0 left
            0.0, 1.0, 0.0, // m0 right
            0.0, 0.0, 1.0, // m1 left
            1.0, 1.0, 1.0, // m1 right
        ];
        let ir = if ir_chunked {
            Dataset::chunked(vec![2, 2, 3], Dtype::F32, vec![1, 2, 3], ir_data).unwrap()
        } else {
            Dataset::contiguous(vec![2, 2, 3], Dtype::F32, ir_data).unwrap()
        };
        file.insert_dataset(DATASET_IR, ir);

        // SourcePosition [M][3]: (az_deg, el_deg, radius_m).
        let positions = vec![
            0.0, 0.0, 1.5, // front
            90.0, 0.0, 1.5, // AES69 left
        ];
        file.insert_dataset(
            DATASET_SOURCE_POSITION,
            Dataset::contiguous(vec![2, 3], Dtype::F64, positions).unwrap(),
        );

        file.insert_dataset(
            DATASET_SAMPLING_RATE,
            Dataset::contiguous(vec![1], Dtype::F64, vec![48_000.0]).unwrap(),
        );
        file
    }

    #[test]
    fn decodes_from_memory_file() {
        let file = fixture_file(false);
        let source = decode_file(&file).unwrap();
        assert_eq!(source.sample_rate(), 48_000);
        assert_eq!(source.convention(), SofaConvention::SimpleFreeFieldHrir);

        let ds = build_dataset(source).unwrap();
        assert_eq!(ds.len(), 2);
        assert_eq!(ds.hrir_len(), 3);
        assert_eq!(ds.sample_rate(), 48_000);
        assert_eq!(ds.left_hrir(0), &[1.0, 0.0, 0.0]);
        assert_eq!(ds.right_hrir(0), &[0.0, 1.0, 0.0]);
        assert_eq!(ds.left_hrir(1), &[0.0, 0.0, 1.0]);
        assert_eq!(ds.right_hrir(1), &[1.0, 1.0, 1.0]);
        // Front stays at zero azimuth; radius carries through.
        assert!(approx(ds.measurement(0).unwrap().azimuth, 0.0, 1e-6));
        assert!(approx(ds.measurement(0).unwrap().distance, 1.5, 1e-6));
        // AES69 left (+90) becomes local -pi/2.
        assert!(approx(ds.measurement(1).unwrap().azimuth, -FRAC_PI_2, 1e-5));
    }

    #[test]
    fn round_trips_through_hdf5_bytes() {
        // Write the fixture to HDF5 bytes, read it back, decode, and compare
        // against an in-memory decode: the byte path must agree numerically.
        let file = fixture_file(false);
        let bytes = file.write();
        let source = decode_sofa(&bytes).unwrap();
        let ds = build_dataset(source).unwrap();

        let direct = build_dataset(decode_file(&file).unwrap()).unwrap();
        assert_eq!(ds.len(), direct.len());
        assert_eq!(ds.hrir_len(), direct.hrir_len());
        assert_eq!(ds.sample_rate(), direct.sample_rate());
        for i in 0..ds.len() {
            assert_eq!(ds.left_hrir(i), direct.left_hrir(i));
            assert_eq!(ds.right_hrir(i), direct.right_hrir(i));
        }
    }

    #[test]
    fn round_trips_with_chunked_ir() {
        let file = fixture_file(true);
        let bytes = file.write();
        let source = decode_sofa(&bytes).unwrap();
        let ds = build_dataset(source).unwrap();
        assert_eq!(ds.len(), 2);
        assert_eq!(ds.left_hrir(1), &[0.0, 0.0, 1.0]);
        assert_eq!(ds.right_hrir(1), &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn rejects_wrong_convention() {
        let mut file = fixture_file(false);
        file.set_attribute(ATTR_SOFA_CONVENTIONS, "GeneralTF");
        let err = decode_file(&file).unwrap_err();
        assert_eq!(
            err,
            DecodeError::UnsupportedConvention("GeneralTF".into())
        );
    }

    #[test]
    fn tolerates_missing_convention_attribute() {
        let mut file = fixture_file(false);
        file.attributes.remove(ATTR_SOFA_CONVENTIONS);
        assert!(decode_file(&file).is_ok());
    }

    #[test]
    fn rejects_missing_ir() {
        let mut file = fixture_file(false);
        file.datasets.remove(DATASET_IR);
        assert_eq!(
            decode_file(&file).unwrap_err(),
            DecodeError::MissingDataset(DATASET_IR)
        );
    }

    #[test]
    fn rejects_missing_source_position() {
        let mut file = fixture_file(false);
        file.datasets.remove(DATASET_SOURCE_POSITION);
        assert_eq!(
            decode_file(&file).unwrap_err(),
            DecodeError::MissingDataset(DATASET_SOURCE_POSITION)
        );
    }

    #[test]
    fn rejects_wrong_receiver_count() {
        let mut file = fixture_file(false);
        // Replace IR with a mono (R=1) tensor.
        file.insert_dataset(
            DATASET_IR,
            Dataset::contiguous(vec![2, 1, 3], Dtype::F32, vec![0.0; 6]).unwrap(),
        );
        assert_eq!(
            decode_file(&file).unwrap_err(),
            DecodeError::BadShape {
                dataset: DATASET_IR
            }
        );
    }

    #[test]
    fn rejects_mismatched_measurement_counts() {
        let mut file = fixture_file(false);
        // SourcePosition with 3 rows but IR has 2 measurements.
        file.insert_dataset(
            DATASET_SOURCE_POSITION,
            Dataset::contiguous(
                vec![3, 3],
                Dtype::F64,
                vec![0.0, 0.0, 1.0, 10.0, 0.0, 1.0, 20.0, 0.0, 1.0],
            )
            .unwrap(),
        );
        assert_eq!(
            decode_file(&file).unwrap_err(),
            DecodeError::InconsistentMeasurementCount {
                ir: 2,
                source_position: 3,
            }
        );
    }

    #[test]
    fn shared_single_source_position_is_broadcast() {
        let mut file = fixture_file(false);
        // One shared position for both measurements.
        file.insert_dataset(
            DATASET_SOURCE_POSITION,
            Dataset::contiguous(vec![1, 3], Dtype::F64, vec![0.0, 30.0, 2.0]).unwrap(),
        );
        let ds = build_dataset(decode_file(&file).unwrap()).unwrap();
        assert_eq!(ds.len(), 2);
        for i in 0..ds.len() {
            assert!(approx(ds.measurement(i).unwrap().distance, 2.0, 1e-6));
            // AES69 elevation +30 preserved as local +30 deg.
            assert!(approx(
                ds.measurement(i).unwrap().elevation,
                30.0 * core::f32::consts::PI / 180.0,
                1e-5
            ));
        }
    }

    #[test]
    fn rejects_invalid_sampling_rate() {
        let mut file = fixture_file(false);
        file.insert_dataset(
            DATASET_SAMPLING_RATE,
            Dataset::contiguous(vec![1], Dtype::F64, vec![0.0]).unwrap(),
        );
        assert_eq!(
            decode_file(&file).unwrap_err(),
            DecodeError::InvalidSamplingRate
        );
    }

    #[test]
    fn maps_h5_error_into_decode_error() {
        // A truncated / bogus byte image should surface as a container error.
        let err = decode_sofa(&[0u8; 4]).unwrap_err();
        matches!(err, DecodeError::Container(_));
        assert!(matches!(err, DecodeError::Container(_)));
    }
}
