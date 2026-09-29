//! Replay-time verification of a recorded capture.
//!
//! [`replay`] takes a [`ReplayPlan`] describing an expected capture and checks
//! that a recorded [`CaptureStream`] is internally consistent and matches the
//! build:
//!
//! * the [`CaptureHeader`] validates against the expected `ABI` digest,
//! * frame indices advance by exactly one with no gaps or regressions,
//! * the deterministic chain digest matches an expected value when supplied,
//!   and
//! * the recorded frame span honours the mode's completeness rule.
//!
//! [`CaptureMode::Crash`] is the one mode that tolerates a short stream: a
//! crash can truncate a capture mid-session, so a stream that stops before the
//! expected final frame is reported as `truncated` rather than an error. All
//! other modes treat a short stream as [`ReplayError::Truncated`], and every
//! mode rejects a stream that runs past the expected final frame.

use core::fmt;

use crate::abi::AbiHash;

use super::header::{self, CaptureHeaderError};
use super::stream::CaptureStream;
use super::{CaptureHeader, CaptureMode};

/// A description of the capture that replay should verify against.
pub struct ReplayPlan<'a> {
    /// Header recorded with the capture.
    pub header: &'a CaptureHeader,
    /// `ABI` digest the current build expects.
    pub expected_abi: &'a AbiHash,
    /// Recorded frame stream to verify.
    pub stream: &'a CaptureStream,
    /// Mode the capture was recorded in; governs truncation tolerance.
    pub mode: CaptureMode,
    /// Optional expected final frame index for completeness checking.
    pub expected_final_index: Option<u64>,
    /// Optional expected chain digest for tamper/reorder detection.
    pub expected_chain_hash: Option<AbiHash>,
}

/// A successful replay verification result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplaySummary {
    /// Number of frames verified.
    pub frame_count: usize,
    /// First recorded frame index, or `None` when the stream is empty.
    pub first_index: Option<u64>,
    /// Last recorded frame index, or `None` when the stream is empty.
    pub last_index: Option<u64>,
    /// Chain digest computed over the verified stream.
    pub chain_hash: AbiHash,
    /// Whether a [`CaptureMode::Crash`] capture stopped before the expected end.
    pub truncated: bool,
}

/// Why replay verification failed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplayError {
    /// The header did not validate against the build/`ABI`.
    Header(CaptureHeaderError),
    /// A frame index failed to advance (duplicate or went backwards).
    NonMonotonic {
        /// Zero-based position of the offending frame.
        position: usize,
        /// Index of the preceding frame.
        previous: u64,
        /// Index recorded at `position`.
        actual: u64,
    },
    /// A frame index skipped one or more values.
    Gap {
        /// Zero-based position of the offending frame.
        position: usize,
        /// Index of the preceding frame.
        previous: u64,
        /// Index recorded at `position`.
        actual: u64,
    },
    /// The stream ran past the expected final frame index.
    Overrun {
        /// Expected final frame index.
        expected_final: u64,
        /// Actual final frame index recorded.
        actual_final: u64,
    },
    /// A non-crash capture ended before the expected final frame index.
    Truncated {
        /// Expected final frame index.
        expected_final: u64,
        /// Actual final frame index, or `None` for an empty stream.
        actual_final: Option<u64>,
    },
    /// The recomputed chain digest did not match the expected digest.
    ChainMismatch {
        /// Expected chain digest.
        expected: AbiHash,
        /// Recomputed chain digest.
        actual: AbiHash,
    },
}

impl fmt::Display for ReplayError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Header(inner) => write!(formatter, "capture header invalid: {inner}"),
            Self::NonMonotonic {
                position,
                previous,
                actual,
            } => write!(
                formatter,
                "frame index regressed at position {position}: previous {previous}, found {actual}"
            ),
            Self::Gap {
                position,
                previous,
                actual,
            } => write!(
                formatter,
                "frame index gap at position {position}: previous {previous}, found {actual}"
            ),
            Self::Overrun {
                expected_final,
                actual_final,
            } => write!(
                formatter,
                "capture overran expected final frame {expected_final}: found {actual_final}"
            ),
            Self::Truncated {
                expected_final,
                actual_final,
            } => match actual_final {
                Some(last) => write!(
                    formatter,
                    "capture truncated before frame {expected_final}: last recorded {last}"
                ),
                None => write!(
                    formatter,
                    "capture truncated before frame {expected_final}: stream is empty"
                ),
            },
            Self::ChainMismatch { expected, actual } => write!(
                formatter,
                "capture chain hash mismatch: expected {expected:?}, found {actual:?}"
            ),
        }
    }
}

impl std::error::Error for ReplayError {}

impl From<CaptureHeaderError> for ReplayError {
    fn from(error: CaptureHeaderError) -> Self {
        Self::Header(error)
    }
}

/// Verifies a capture against `plan`.
///
/// On success returns a [`ReplaySummary`] describing the verified span and the
/// recomputed chain digest.
///
/// # Errors
///
/// Returns a [`ReplayError`] describing the first inconsistency found: header
/// validation failures, frame continuity breaks (with the offending position),
/// completeness violations, or a chain-digest mismatch.
pub fn replay(plan: &ReplayPlan<'_>) -> Result<ReplaySummary, ReplayError> {
    header::validate(plan.header, plan.expected_abi)?;

    let frames = plan.stream.frames();
    let mut previous: Option<u64> = None;
    for (position, frame) in frames.iter().enumerate() {
        if let Some(prev) = previous {
            let actual = frame.frame_index;
            if actual <= prev {
                return Err(ReplayError::NonMonotonic {
                    position,
                    previous: prev,
                    actual,
                });
            }
            if actual != prev + 1 {
                return Err(ReplayError::Gap {
                    position,
                    previous: prev,
                    actual,
                });
            }
        }
        previous = Some(frame.frame_index);
    }

    let chain_hash = plan.stream.chain_hash(plan.header.random_seed);
    if let Some(expected) = plan.expected_chain_hash
        && chain_hash != expected
    {
        return Err(ReplayError::ChainMismatch {
            expected,
            actual: chain_hash,
        });
    }

    let last_index = plan.stream.last_frame_index();
    let mut truncated = false;
    if let Some(expected_final) = plan.expected_final_index {
        match last_index {
            Some(last) if last > expected_final => {
                return Err(ReplayError::Overrun {
                    expected_final,
                    actual_final: last,
                });
            }
            Some(last) if last == expected_final => {}
            incomplete => {
                if matches!(plan.mode, CaptureMode::Crash) {
                    truncated = true;
                } else {
                    return Err(ReplayError::Truncated {
                        expected_final,
                        actual_final: incomplete,
                    });
                }
            }
        }
    }

    Ok(ReplaySummary {
        frame_count: frames.len(),
        first_index: plan.stream.first_frame_index(),
        last_index,
        chain_hash,
        truncated,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::stream::FrameRecord;
    use crate::gpu_scene::GpuSceneSnapshot;

    const SEED: u64 = 0xDEAD_BEEF;

    fn abi() -> AbiHash {
        AbiHash::from_bytes(b"replay-contract")
    }

    fn header() -> CaptureHeader {
        CaptureHeader {
            architecture_version: crate::ARCHITECTURE_VERSION,
            abi_hash: abi(),
            scene: GpuSceneSnapshot::default(),
            random_seed: SEED,
        }
    }

    fn stream_of(indices: &[u64]) -> CaptureStream {
        indices.iter().map(|&i| FrameRecord::new(i)).collect()
    }

    fn plan<'a>(
        header: &'a CaptureHeader,
        expected_abi: &'a AbiHash,
        stream: &'a CaptureStream,
        mode: CaptureMode,
    ) -> ReplayPlan<'a> {
        ReplayPlan {
            header,
            expected_abi,
            stream,
            mode,
            expected_final_index: None,
            expected_chain_hash: None,
        }
    }

    #[test]
    fn empty_stream_replays_cleanly() {
        let header = header();
        let expected = abi();
        let stream = CaptureStream::new();
        let summary = replay(&plan(&header, &expected, &stream, CaptureMode::Quality)).unwrap();
        assert_eq!(summary.frame_count, 0);
        assert_eq!(summary.first_index, None);
        assert_eq!(summary.last_index, None);
        assert!(!summary.truncated);
        assert_eq!(summary.chain_hash, stream.chain_hash(SEED));
    }

    #[test]
    fn contiguous_stream_replays() {
        let header = header();
        let expected = abi();
        let stream = stream_of(&[0, 1, 2, 3]);
        let summary = replay(&plan(&header, &expected, &stream, CaptureMode::Performance)).unwrap();
        assert_eq!(summary.frame_count, 4);
        assert_eq!(summary.first_index, Some(0));
        assert_eq!(summary.last_index, Some(3));
    }

    #[test]
    fn header_error_propagates() {
        let header = header();
        let wrong = AbiHash::from_bytes(b"other-contract");
        let stream = stream_of(&[0, 1]);
        let error = replay(&plan(&header, &wrong, &stream, CaptureMode::Quality)).unwrap_err();
        assert!(matches!(error, ReplayError::Header(_)));
    }

    #[test]
    fn gap_is_reported_with_position() {
        let header = header();
        let expected = abi();
        let stream = stream_of(&[0, 1, 3]);
        let error = replay(&plan(&header, &expected, &stream, CaptureMode::Quality)).unwrap_err();
        assert_eq!(
            error,
            ReplayError::Gap {
                position: 2,
                previous: 1,
                actual: 3,
            }
        );
    }

    #[test]
    fn regression_is_non_monotonic() {
        let header = header();
        let expected = abi();
        let stream = stream_of(&[0, 1, 1]);
        let error = replay(&plan(&header, &expected, &stream, CaptureMode::Quality)).unwrap_err();
        assert_eq!(
            error,
            ReplayError::NonMonotonic {
                position: 2,
                previous: 1,
                actual: 1,
            }
        );
    }

    #[test]
    fn overrun_is_rejected_in_every_mode() {
        let header = header();
        let expected = abi();
        let stream = stream_of(&[0, 1, 2]);
        for mode in [
            CaptureMode::Quality,
            CaptureMode::Performance,
            CaptureMode::Crash,
        ] {
            let mut request = plan(&header, &expected, &stream, mode);
            request.expected_final_index = Some(1);
            let error = replay(&request).unwrap_err();
            assert_eq!(
                error,
                ReplayError::Overrun {
                    expected_final: 1,
                    actual_final: 2,
                }
            );
        }
    }

    #[test]
    fn truncation_is_rejected_outside_crash_mode() {
        let header = header();
        let expected = abi();
        let stream = stream_of(&[0, 1]);
        let mut request = plan(&header, &expected, &stream, CaptureMode::Quality);
        request.expected_final_index = Some(4);
        let error = replay(&request).unwrap_err();
        assert_eq!(
            error,
            ReplayError::Truncated {
                expected_final: 4,
                actual_final: Some(1),
            }
        );
    }

    #[test]
    fn truncation_is_tolerated_in_crash_mode() {
        let header = header();
        let expected = abi();
        let stream = stream_of(&[0, 1]);
        let mut request = plan(&header, &expected, &stream, CaptureMode::Crash);
        request.expected_final_index = Some(4);
        let summary = replay(&request).unwrap();
        assert!(summary.truncated);
        assert_eq!(summary.last_index, Some(1));
    }

    #[test]
    fn empty_crash_capture_is_truncated_not_error() {
        let header = header();
        let expected = abi();
        let stream = CaptureStream::new();
        let mut request = plan(&header, &expected, &stream, CaptureMode::Crash);
        request.expected_final_index = Some(2);
        let summary = replay(&request).unwrap();
        assert!(summary.truncated);
        assert_eq!(summary.last_index, None);
    }

    #[test]
    fn empty_non_crash_with_expected_is_truncated_error() {
        let header = header();
        let expected = abi();
        let stream = CaptureStream::new();
        let mut request = plan(&header, &expected, &stream, CaptureMode::Quality);
        request.expected_final_index = Some(2);
        let error = replay(&request).unwrap_err();
        assert_eq!(
            error,
            ReplayError::Truncated {
                expected_final: 2,
                actual_final: None,
            }
        );
    }

    #[test]
    fn exact_final_index_passes() {
        let header = header();
        let expected = abi();
        let stream = stream_of(&[0, 1, 2]);
        let mut request = plan(&header, &expected, &stream, CaptureMode::Quality);
        request.expected_final_index = Some(2);
        let summary = replay(&request).unwrap();
        assert!(!summary.truncated);
    }

    #[test]
    fn matching_chain_hash_passes() {
        let header = header();
        let expected = abi();
        let stream = stream_of(&[0, 1, 2]);
        let good = stream.chain_hash(SEED);
        let mut request = plan(&header, &expected, &stream, CaptureMode::Quality);
        request.expected_chain_hash = Some(good);
        assert!(replay(&request).is_ok());
    }

    #[test]
    fn chain_hash_mismatch_is_detected() {
        let header = header();
        let expected = abi();
        let stream = stream_of(&[0, 1, 2]);
        let wrong = AbiHash::from_bytes(b"not-the-chain");
        let mut request = plan(&header, &expected, &stream, CaptureMode::Quality);
        request.expected_chain_hash = Some(wrong);
        let error = replay(&request).unwrap_err();
        assert_eq!(
            error,
            ReplayError::ChainMismatch {
                expected: wrong,
                actual: stream.chain_hash(SEED),
            }
        );
    }
}
