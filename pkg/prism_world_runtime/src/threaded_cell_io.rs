//! Fixed-worker archive routing for frame-driven cell loading.

use crate::{IoByteLimits, RoutedCellIo, ScheduledCellIo};
use prism_world_io::{QueuedThreadedFileProvider, ThreadedFileConfig, ThreadedFileProvider};
use std::io;
use std::path::PathBuf;

/// Manifest-checked asynchronous archive reads on a fixed filesystem worker pool.
pub type ThreadedCellIo = RoutedCellIo<ScheduledCellIo<QueuedThreadedFileProvider>>;

impl ThreadedCellIo {
    /// Creates a routed worker backend with explicit request and byte bounds.
    ///
    /// `max_in_flight` must not exceed `workers.capacity`. This reserves enough
    /// admission records for replacement requests while canceled reads finish
    /// worker cleanup. The pool is exclusive to this backend. Chunk size is
    /// `workers.max_chunk_size`; all file operations remain on its workers.
    ///
    /// Registration does not read the file. Use [`Self::register`] before passing
    /// the backend to [`crate::CellFrameCoordinator::with_io`].
    pub fn threaded(
        path: impl Into<PathBuf>,
        workers: ThreadedFileConfig,
        max_in_flight: usize,
        capacity: usize,
        limits: IoByteLimits,
    ) -> io::Result<Self> {
        if max_in_flight == 0 || max_in_flight > workers.capacity || capacity == 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidInput,
                "active IO limit must be nonzero and fit worker capacity; request capacity must be nonzero"));
        }
        let provider = ThreadedFileProvider::new(path, workers)?.queued();
        let io = ScheduledCellIo::with_byte_limits(
            provider,
            max_in_flight,
            capacity,
            workers.max_chunk_size,
            limits,
        )
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error.to_string()))?;
        Ok(Self::new(io))
    }
}
