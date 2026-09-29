//! Capture-header validation and its detailed error taxonomy.
//!
//! Before a capture can be trusted for replay its [`CaptureHeader`] must be
//! validated against the running build. Two invariants matter:
//!
//! * the capture's `architecture_version` must equal the crate's
//!   [`crate::ARCHITECTURE_VERSION`], and
//! * the capture's [`AbiHash`] must equal the `ABI` digest the caller expects
//!   for the current shader/data contracts.
//!
//! Both checks report the expected and actual values so a mismatch is
//! actionable rather than a bare boolean failure.

use core::fmt;

use crate::abi::AbiHash;

use super::CaptureHeader;

/// Why a [`CaptureHeader`] failed validation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CaptureHeaderError {
    /// The capture's architecture version does not match this build.
    VersionMismatch {
        /// Version this build expects (`crate::ARCHITECTURE_VERSION`).
        expected: u32,
        /// Version recorded in the capture header.
        actual: u32,
    },
    /// The capture's `ABI` digest does not match the expected contract digest.
    AbiMismatch {
        /// `ABI` digest the caller expects for the current contracts.
        expected: AbiHash,
        /// `ABI` digest recorded in the capture header.
        actual: AbiHash,
    },
}

impl fmt::Display for CaptureHeaderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::VersionMismatch { expected, actual } => write!(
                formatter,
                "capture architecture version mismatch: expected {expected}, found {actual}"
            ),
            Self::AbiMismatch { expected, actual } => write!(
                formatter,
                "capture ABI hash mismatch: expected {expected:?}, found {actual:?}"
            ),
        }
    }
}

impl std::error::Error for CaptureHeaderError {}

/// Validates `header` against this build and the expected `ABI` digest.
///
/// Returns `Ok(())` when the architecture version matches
/// [`crate::ARCHITECTURE_VERSION`] and the header's `abi_hash` equals
/// `expected_abi`. The version check runs first so a stale capture reports the
/// version gap before the (necessarily different) `ABI` gap.
///
/// # Errors
///
/// Returns [`CaptureHeaderError::VersionMismatch`] or
/// [`CaptureHeaderError::AbiMismatch`] with the expected and actual values.
pub fn validate(header: &CaptureHeader, expected_abi: &AbiHash) -> Result<(), CaptureHeaderError> {
    if header.architecture_version != crate::ARCHITECTURE_VERSION {
        return Err(CaptureHeaderError::VersionMismatch {
            expected: crate::ARCHITECTURE_VERSION,
            actual: header.architecture_version,
        });
    }
    if header.abi_hash != *expected_abi {
        return Err(CaptureHeaderError::AbiMismatch {
            expected: *expected_abi,
            actual: header.abi_hash,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_scene::GpuSceneSnapshot;

    fn header_with(version: u32, abi_hash: AbiHash) -> CaptureHeader {
        CaptureHeader {
            architecture_version: version,
            abi_hash,
            scene: GpuSceneSnapshot::default(),
            random_seed: 0,
        }
    }

    #[test]
    fn valid_header_passes() {
        let abi = AbiHash::from_bytes(b"contract-v1");
        let header = header_with(crate::ARCHITECTURE_VERSION, abi);
        assert_eq!(validate(&header, &abi), Ok(()));
    }

    #[test]
    fn version_mismatch_is_detailed() {
        let abi = AbiHash::from_bytes(b"contract-v1");
        let header = header_with(crate::ARCHITECTURE_VERSION + 1, abi);
        let error = validate(&header, &abi).unwrap_err();
        assert_eq!(
            error,
            CaptureHeaderError::VersionMismatch {
                expected: crate::ARCHITECTURE_VERSION,
                actual: crate::ARCHITECTURE_VERSION + 1,
            }
        );
    }

    #[test]
    fn abi_mismatch_is_detailed() {
        let expected = AbiHash::from_bytes(b"contract-v1");
        let actual = AbiHash::from_bytes(b"contract-v2");
        let header = header_with(crate::ARCHITECTURE_VERSION, actual);
        let error = validate(&header, &expected).unwrap_err();
        assert_eq!(error, CaptureHeaderError::AbiMismatch { expected, actual });
    }

    #[test]
    fn version_is_checked_before_abi() {
        // Both fields wrong: the version gap must be reported first.
        let expected = AbiHash::from_bytes(b"contract-v1");
        let actual = AbiHash::from_bytes(b"contract-v2");
        let header = header_with(crate::ARCHITECTURE_VERSION + 9, actual);
        match validate(&header, &expected) {
            Err(CaptureHeaderError::VersionMismatch { .. }) => {}
            other => panic!("expected version mismatch first, got {other:?}"),
        }
    }

    #[test]
    fn display_mentions_both_sides() {
        let error = CaptureHeaderError::VersionMismatch {
            expected: 1,
            actual: 2,
        };
        let text = alloc::format!("{error}");
        assert!(text.contains('1'));
        assert!(text.contains('2'));
    }
}
