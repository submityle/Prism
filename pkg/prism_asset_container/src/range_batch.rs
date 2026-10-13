//! Deterministic batching for runtime payload range reads.
//!
//! The batcher operates on the existing `ContainerReader`/`RangeSource`
//! contract. It validates every logical request first, sorts only the physical
//! read plan, merges nearby or overlapping ranges within explicit limits, and
//! restores results in caller order. It is a range/chunk scheduling primitive,
//! not a codec, GPU, or publication protocol.

use prism_asset::StableGuid;

use crate::ContainerError;

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
    /// Maximum size of every physical read, including gaps between requests.
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

impl RangeBatchPolicy {
    /// Validates the policy before any request is submitted to a source.
    pub(crate) fn validate(self) -> Result<(), ContainerError> {
        if self.max_batch_bytes == 0 {
            return Err(ContainerError::InvalidBatchPolicy);
        }
        Ok(())
    }

    /// Validates that one request can be represented by a bounded physical
    /// read under this policy.
    pub(crate) fn validate_request_len(self, len: u64) -> Result<(), ContainerError> {
        self.validate()?;
        if len > self.max_batch_bytes {
            return Err(ContainerError::InvalidBatchPolicy);
        }
        Ok(())
    }
}

/// Computes an exclusive range end without allowing integer wraparound.
pub(crate) fn checked_range_end(offset: u64, len: u64) -> Result<u64, ContainerError> {
    offset
        .checked_add(len)
        .ok_or(ContainerError::IntegerOverflow)
}

/// Computes a physical range from a container offset and a payload-local
/// range while checking both additions independently.
pub(crate) fn checked_physical_range(
    base: u64,
    offset: u64,
    len: u64,
) -> Result<(u64, u64), ContainerError> {
    let local_end = checked_range_end(offset, len)?;
    let start = base
        .checked_add(offset)
        .ok_or(ContainerError::IntegerOverflow)?;
    let end = base
        .checked_add(local_end)
        .ok_or(ContainerError::IntegerOverflow)?;
    Ok((start, end))
}

#[cfg(test)]
mod tests {
    use super::{checked_physical_range, checked_range_end, RangeBatchPolicy};
    use crate::ContainerError;

    #[test]
    fn checked_ranges_reject_both_local_and_physical_overflow() {
        assert_eq!(
            checked_range_end(u64::MAX, 1),
            Err(ContainerError::IntegerOverflow)
        );
        assert_eq!(
            checked_physical_range(u64::MAX, 1, 0),
            Err(ContainerError::IntegerOverflow)
        );
        assert_eq!(
            checked_physical_range(10, u64::MAX, 1),
            Err(ContainerError::IntegerOverflow)
        );
    }

    #[test]
    fn policy_rejects_zero_and_unrepresentable_request_bounds() {
        assert_eq!(
            (RangeBatchPolicy {
                max_batch_bytes: 0,
                max_gap_bytes: 0,
            })
            .validate(),
            Err(ContainerError::InvalidBatchPolicy)
        );
        assert_eq!(
            (RangeBatchPolicy {
                max_batch_bytes: 4,
                max_gap_bytes: 8,
            })
            .validate_request_len(5),
            Err(ContainerError::InvalidBatchPolicy)
        );
        assert!((RangeBatchPolicy {
            max_batch_bytes: 4,
            max_gap_bytes: 8,
        })
        .validate_request_len(4)
        .is_ok());
    }
}
