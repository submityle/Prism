//! Stable file/byte-range routing without an asset-server dependency.

use alloc::collections::BTreeMap;
use alloc::format;
use std::path::PathBuf;

use prism_world_io::{ConfigError, FileProvider, IoError};
use prism_world_stream::SectorId;

use crate::{CancellationToken, CellReadRequest, ChunkProvider, IoByteLimits, IoSchedulerAdapter};

/// Routes cell manifests to validated ranges in one file or packed cell archive.
///
/// Each source records a complete manifest, byte offset, and optional FNV-1a
/// checksum. A changed manifest must be registered explicitly: stale versions
/// cannot silently read a previous cell's range. Files are opened independently
/// by the existing [`FileProvider`] for each scheduler operation.
pub struct FileCellProvider {
    adapter: IoSchedulerAdapter<FileProvider>,
    sources: BTreeMap<SectorId, CellReadRequest>,
}

impl FileCellProvider {
    /// Creates a file provider with the scheduler's default 64 KiB chunk size.
    pub fn new(
        path: impl Into<PathBuf>,
        max_in_flight: usize,
        capacity: usize,
    ) -> Result<Self, ConfigError> {
        Self::with_chunk_size(path, max_in_flight, capacity, 64 * 1024)
    }

    /// Creates a file provider with explicit per-poll chunk size and queue bounds.
    pub fn with_chunk_size(
        path: impl Into<PathBuf>,
        max_in_flight: usize,
        capacity: usize,
        chunk_size: usize,
    ) -> Result<Self, ConfigError> {
        Ok(Self {
            adapter: IoSchedulerAdapter::with_chunk_size(
                FileProvider::new(path),
                max_in_flight,
                capacity,
                chunk_size,
            )?,
            sources: BTreeMap::new(),
        })
    }

    /// Creates a file provider with explicit declared-byte admission limits.
    ///
    /// Limits are enforced on resolved requests, not on source registration.
    /// No byte limit is silently clamped. See [`IoByteLimits`] for memory scope.
    pub fn with_byte_limits(
        path: impl Into<PathBuf>,
        max_in_flight: usize,
        capacity: usize,
        chunk_size: usize,
        limits: IoByteLimits,
    ) -> Result<Self, ConfigError> {
        Ok(Self {
            adapter: IoSchedulerAdapter::with_byte_limits(
                FileProvider::new(path),
                max_in_flight,
                capacity,
                chunk_size,
                limits,
            )?,
            sources: BTreeMap::new(),
        })
    }

    /// Registers or replaces the file range for a cell without opening the file.
    ///
    /// The request length is its manifest's `io_bytes`, including the fixed header.
    /// An overflowing range is rejected without changing the previous registration.
    pub fn register(
        &mut self,
        source: CellReadRequest,
    ) -> Result<Option<CellReadRequest>, IoError> {
        let range = source.io_request();
        range
            .offset()
            .checked_add(range.length())
            .ok_or(IoError::RangeOverflow)?;
        Ok(self.sources.insert(source.sector(), source))
    }

    /// Resolves a current manifest to its file range, preserving the caller's priority.
    pub fn resolve(&self, request: &CellReadRequest) -> Result<CellReadRequest, IoError> {
        let Some(source) = self.sources.get(&request.sector()) else {
            return Err(IoError::Provider {
                message: format!("no file range registered for sector {:?}", request.sector()),
            });
        };
        if source.manifest() != request.manifest() {
            return Err(IoError::Provider {
                message: format!("stale file range for sector {:?}", request.sector()),
            });
        }
        let mut resolved = *source;
        resolved.priority = request.priority();
        Ok(resolved)
    }

    /// Returns the shared chunk scheduler for explicit asynchronous polling.
    ///
    /// Pass [`Self::resolve`]'s result to `submit` to retain source range/checksum
    /// validation. The [`ChunkProvider`] implementation does this automatically.
    #[must_use]
    pub const fn scheduler(&self) -> &IoSchedulerAdapter<FileProvider> {
        &self.adapter
    }
}

impl ChunkProvider for FileCellProvider {
    type Error = IoError;

    fn read_chunk(
        &self,
        request: &CellReadRequest,
        cancel: CancellationToken,
    ) -> Result<alloc::vec::Vec<u8>, Self::Error> {
        self.adapter
            .read_to_completion(&self.resolve(request)?, cancel)
    }
}
