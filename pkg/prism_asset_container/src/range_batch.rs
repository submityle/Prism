//! Deterministic batching for runtime payload range reads.
//!
//! The batcher operates on the existing `ContainerReader`/`RangeSource`
//! contract. It validates every logical request first, sorts only the physical
//! read plan, merges nearby or overlapping ranges within explicit limits, and
//! restores results in caller order. It is a range/chunk scheduling primitive,
//! not a codec, GPU, or publication protocol.

use prism_asset::StableGuid;

/// One logical range request within a runtime container.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RangeRequest {
    /// Container-local payload identity.
    pub guid: StableGuid,
    /// Offset within the decoded payload.
    pub offset: u64,
    /// Number of bytes requested.
    pub len: u64,
}

impl RangeRequest {
    /// Creates a payload range request.
    #[must_use]
    pub const fn new(guid: StableGuid, offset: u64, len: u64) -> Self {
        Self { guid, offset, len }
    }
}

/// Bounds used when coalescing physical range reads.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RangeBatchPolicy {
    /// Maximum size of one physical read, including gaps between requests.
    pub max_batch_bytes: u64,
    /// Maximum gap that may be read between adjacent requests.
    pub max_gap_bytes: u64,
}

impl Default for RangeBatchPolicy {
    fn default() -> Self {
        Self {
            max_batch_bytes: 1024 * 1024,
            max_gap_bytes: 4096,
        }
    }
}
