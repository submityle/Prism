//! Persistent GPU storage for the virtual-texture indirection page table.
//!
//! The CPU page table remains the source of truth. This object owns one
//! bounded, copyable storage buffer and replaces its contents deterministically
//! on every upload, including zeroing unused capacity.

use core::fmt;

use prism_render_architecture::texture_streaming::{GpuPageTable, PAGE_TABLE_ENTRY_WORDS};
use wgpu::Buffer;

use crate::{buffer, context::GpuContext};

const WORD_BYTES: usize = size_of::<u32>();

#[derive(Clone, Debug, Eq, PartialEq)]
/// Errors raised while creating or updating persistent page-table storage.
pub enum PageTableStorageError {
    /// The storage object was requested with zero entry capacity.
    ZeroCapacity,
    /// The upload contains more entries than the fixed storage capacity.
    CapacityExceeded {
        /// Number of entries in the rejected upload.
        requested: usize,
        /// Fixed number of entries supported by the storage object.
        capacity: usize,
    },
    /// The upload is not a whole number of page-table entries.
    InvalidWordCount {
        /// Number of words in the rejected upload.
        words: usize,
    },
    /// The upload uses the reserved physical-slot miss sentinel.
    ReservedMissSentinel {
        /// Zero-based entry index containing the sentinel.
        entry: usize,
    },
    /// The requested allocation exceeds the device storage binding limit.
    DeviceBufferLimitExceeded {
        /// Requested allocation size in bytes.
        requested_bytes: u64,
        /// Device-reported maximum storage binding size in bytes.
        limit_bytes: u64,
    },
    /// An intermediate byte-count conversion overflowed.
    ByteCountOverflow,
}

impl fmt::Display for PageTableStorageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroCapacity => formatter.write_str("page-table storage capacity must be non-zero"),
            Self::CapacityExceeded { requested, capacity } => write!(formatter, "page-table has {requested} entries but storage capacity is {capacity}"),
            Self::InvalidWordCount { words } => write!(formatter, "page-table word count {words} is not divisible by {PAGE_TABLE_ENTRY_WORDS}"),
            Self::ReservedMissSentinel { entry } => write!(formatter, "page-table entry {entry} uses the reserved u32::MAX miss sentinel"),
            Self::DeviceBufferLimitExceeded { requested_bytes, limit_bytes } => write!(formatter, "page-table storage requires {requested_bytes} bytes but device limit is {limit_bytes}"),
            Self::ByteCountOverflow => formatter.write_str("page-table storage byte count overflowed"),
        }
    }
}

impl std::error::Error for PageTableStorageError {}

/// Persistent, bounded GPU storage for the canonical page-table word layout.
pub struct GpuPageTableStorage {
    buffer: Buffer,
    capacity_entries: usize,
    len: usize,
}

impl GpuPageTableStorage {
    /// Allocates a zeroed storage buffer for a fixed entry capacity.
    pub fn new(ctx: &GpuContext, capacity_entries: usize) -> Result<Self, PageTableStorageError> {
        if capacity_entries == 0 {
            return Err(PageTableStorageError::ZeroCapacity);
        }
        let capacity_words = capacity_entries
            .checked_mul(PAGE_TABLE_ENTRY_WORDS)
            .ok_or(PageTableStorageError::ByteCountOverflow)?;
        let capacity_bytes = capacity_words
            .checked_mul(WORD_BYTES)
            .ok_or(PageTableStorageError::ByteCountOverflow)?;
        let requested_bytes =
            u64::try_from(capacity_bytes).map_err(|_| PageTableStorageError::ByteCountOverflow)?;
        let limit_bytes = ctx.device().limits().max_storage_buffer_binding_size;
        if requested_bytes > limit_bytes {
            return Err(PageTableStorageError::DeviceBufferLimitExceeded {
                requested_bytes,
                limit_bytes,
            });
        }
        Ok(Self {
            buffer: buffer::storage_rw_zeroed(
                ctx.device(),
                "prism_vt_page_table_storage",
                requested_bytes,
            ),
            capacity_entries,
            len: 0,
        })
    }

    /// Uploads a CPU page table and clears all unused capacity.
    pub fn upload(
        &mut self,
        ctx: &GpuContext,
        table: &GpuPageTable,
    ) -> Result<(), PageTableStorageError> {
        self.upload_words(ctx, table.words())
    }

    /// Uploads canonical flat page-table words and clears all unused capacity.
    pub fn upload_words(
        &mut self,
        ctx: &GpuContext,
        words: &[u32],
    ) -> Result<(), PageTableStorageError> {
        let entry_count = validate_words(words)?;
        if entry_count > self.capacity_entries {
            return Err(PageTableStorageError::CapacityExceeded {
                requested: entry_count,
                capacity: self.capacity_entries,
            });
        }
        let capacity_words = self
            .capacity_entries
            .checked_mul(PAGE_TABLE_ENTRY_WORDS)
            .ok_or(PageTableStorageError::ByteCountOverflow)?;
        let mut padded = vec![0u32; capacity_words];
        padded[..words.len()].copy_from_slice(words);
        ctx.queue()
            .write_buffer(&self.buffer, 0, bytemuck::cast_slice(&padded));
        self.len = entry_count;
        Ok(())
    }

    #[must_use]
    /// Returns the persistent storage buffer for shader binding or GPU copies.
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    #[must_use]
    /// Returns the logical entry count from the last successful upload.
    pub const fn len(&self) -> usize {
        self.len
    }

    #[must_use]
    /// Returns whether the last successful upload was an empty table.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[must_use]
    /// Returns the fixed entry capacity.
    pub const fn capacity_entries(&self) -> usize {
        self.capacity_entries
    }
}

fn validate_words(words: &[u32]) -> Result<usize, PageTableStorageError> {
    if !words.len().is_multiple_of(PAGE_TABLE_ENTRY_WORDS) {
        return Err(PageTableStorageError::InvalidWordCount { words: words.len() });
    }
    for (entry, chunk) in words.chunks_exact(PAGE_TABLE_ENTRY_WORDS).enumerate() {
        if chunk[PAGE_TABLE_ENTRY_WORDS - 1] == u32::MAX {
            return Err(PageTableStorageError::ReservedMissSentinel { entry });
        }
    }
    Ok(words.len() / PAGE_TABLE_ENTRY_WORDS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_empty_table_words() {
        assert_eq!(validate_words(&[]), Ok(0));
    }

    #[test]
    fn rejects_partial_and_reserved_words() {
        assert_eq!(
            validate_words(&[1, 2, 3]),
            Err(PageTableStorageError::InvalidWordCount { words: 3 })
        );
        assert_eq!(
            validate_words(&[1, 2, 3, u32::MAX]),
            Err(PageTableStorageError::ReservedMissSentinel { entry: 0 })
        );
    }
}
