//! Deterministic render capture and replay metadata.
//!
//! This module models the `CPU`-verifiable half of Prism's render capture
//! system. A capture is a [`CaptureHeader`] plus an ordered
//! [`CaptureStream`] of [`FrameRecord`]s. Everything here is deterministic and
//! `GPU`-free so captures can be recorded, hashed, and replayed on any host,
//! including sandboxes without a graphics device.
//!
//! The pieces fit together as follows:
//!
//! * [`header::validate`] checks a [`CaptureHeader`] against this build's
//!   [`crate::ARCHITECTURE_VERSION`] and an expected [`AbiHash`], reporting a
//!   detailed [`CaptureHeaderError`] on mismatch.
//! * [`stream`] holds the append-only [`FrameRecord`] sequence and its
//!   deterministic byte encoding.
//! * [`hash_chain`] folds frames into a rolling [`FrameHashChain`] digest that
//!   is order- and byte-sensitive, reusing the crate's `FNV`-1a mixing.
//! * [`replay`] verifies a recorded stream end to end: header, frame
//!   continuity, mode-specific completeness, and chain digest.
//!
//! Fields such as [`FrameRecord::command_digest`] and
//! [`FrameRecord::gpu_duration_ns`] carry `GPU`-sourced data once the render
//! backend exists; until then they hold caller-supplied deterministic values
//! and are treated as opaque bytes.

use crate::{abi::AbiHash, gpu_scene::GpuSceneSnapshot};

pub mod hash_chain;
pub mod header;
pub mod replay;
pub mod stream;

pub use hash_chain::FrameHashChain;
pub use header::{validate, CaptureHeaderError};
pub use replay::{replay, ReplayError, ReplayPlan, ReplaySummary};
pub use stream::{CaptureStream, FrameRecord};

/// Identifying metadata recorded at the head of a capture.
///
/// The header binds a capture to a specific build (`architecture_version`),
/// data/shader contract (`abi_hash`), scene snapshot, and the random seed used
/// to drive deterministic per-frame hashing.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CaptureHeader {
    /// Architecture contract version the capture was recorded against.
    pub architecture_version: u32,
    /// `ABI` digest of the shader/data contracts in effect at capture time.
    pub abi_hash: AbiHash,
    /// Snapshot of the `GPU` scene at the start of the capture.
    pub scene: GpuSceneSnapshot,
    /// Seed used to initialize the deterministic capture hash chain.
    pub random_seed: u64,
}

impl CaptureHeader {
    /// Validates this header against `expected_abi` and the running build.
    ///
    /// This is a convenience wrapper over [`header::validate`].
    ///
    /// # Errors
    ///
    /// Returns a [`CaptureHeaderError`] describing the version or `ABI`
    /// mismatch.
    pub fn validate(&self, expected_abi: &AbiHash) -> Result<(), CaptureHeaderError> {
        validate(self, expected_abi)
    }
}

/// The fidelity/lifecycle mode a capture was recorded in.
///
/// The mode governs how strict replay completeness checking is: a
/// [`CaptureMode::Crash`] capture may legitimately be truncated, whereas the
/// other modes are expected to run to their declared final frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CaptureMode {
    /// Full-fidelity capture; a complete, untruncated stream is expected.
    Quality,
    /// Lower-overhead capture; still expected to be complete.
    Performance,
    /// Best-effort capture that may be truncated by a crash.
    Crash,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_scene::GpuSceneSnapshot;

    #[test]
    fn end_to_end_capture_replays() {
        let abi = AbiHash::from_bytes(b"integration-contract");
        let header = CaptureHeader {
            architecture_version: crate::ARCHITECTURE_VERSION,
            abi_hash: abi,
            scene: GpuSceneSnapshot::default(),
            random_seed: 0x0BAD_F00D,
        };

        let mut stream = CaptureStream::new();
        stream.push(FrameRecord::new(0).with_draw_call_count(12));
        stream.push(FrameRecord::new(1).with_command_digest(alloc::vec![1u8, 2, 3]));
        stream.push(FrameRecord::new(2).with_gpu_duration_ns(9_000));

        let chain = stream.chain_hash(header.random_seed);
        let request = ReplayPlan {
            header: &header,
            expected_abi: &abi,
            stream: &stream,
            mode: CaptureMode::Quality,
            expected_final_index: Some(2),
            expected_chain_hash: Some(chain),
        };

        let summary = replay(&request).expect("capture should replay");
        assert_eq!(summary.frame_count, 3);
        assert_eq!(summary.last_index, Some(2));
        assert_eq!(summary.chain_hash, chain);
        assert!(!summary.truncated);
    }

    #[test]
    fn header_convenience_validate_matches_free_function() {
        let abi = AbiHash::from_bytes(b"wrapper-contract");
        let header = CaptureHeader {
            architecture_version: crate::ARCHITECTURE_VERSION,
            abi_hash: abi,
            scene: GpuSceneSnapshot::default(),
            random_seed: 1,
        };
        assert_eq!(header.validate(&abi), validate(&header, &abi));
    }

    #[test]
    fn capture_modes_are_distinct() {
        assert_ne!(CaptureMode::Quality, CaptureMode::Performance);
        assert_ne!(CaptureMode::Performance, CaptureMode::Crash);
        assert_ne!(CaptureMode::Quality, CaptureMode::Crash);
    }
}
