//! Queue-ordered, bounded asynchronous buffer snapshots.

use crate::BufferId;
use alloc::vec::Vec;

/// An exact byte range captured at the queue position of a readback request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferReadbackRange {
    /// Source buffer, which must support copy-source usage.
    pub buffer: BufferId,
    /// Byte offset, aligned to the backend's buffer-copy alignment.
    pub offset: u64,
    /// Nonzero byte count, aligned to the buffer-copy alignment.
    pub size: u64,
}

/// Failure to admit or complete an asynchronous snapshot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferReadbackError {
    /// The backend does not implement asynchronous readback.
    Unsupported,
    /// The range is empty, unaligned, overflowing, or outside its source.
    InvalidRange {
        /// Index in the original request.
        index: usize,
    },
    /// Packed staging allocation would exceed the explicit request budget.
    BudgetExceeded {
        /// Required staging bytes, including packing alignment.
        requested: u64,
        /// Caller-provided allocation limit.
        limit: u64,
    },
    /// A source id is stale, missing, or has incompatible usage.
    ResourceUnavailable,
    /// The source device has been lost; this ticket cannot recover.
    DeviceLost,
    /// The driver rejected the staging mapping.
    MapFailed,
    /// Completion bytes have already been taken from this ticket.
    Consumed,
}

/// Owns one immutable queue-position snapshot until completion or cancellation.
///
/// Implementations retain the required native resources independently of RHI
/// source ids. Polling never waits for GPU completion. Dropping a ticket cancels
/// delivery without requiring device idle. It does not undo the submitted copy.
pub trait BufferReadback: Send + Sync {
    /// Returns `None` while pending, or each requested range's exact owned bytes
    /// in request order. Successful delivery consumes the ticket; subsequent
    /// calls return [`BufferReadbackError::Consumed`].
    fn try_complete(&mut self) -> Result<Option<Vec<Vec<u8>>>, BufferReadbackError>;
}
