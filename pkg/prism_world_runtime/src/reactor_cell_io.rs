//! Native archive reads borrowing the host's existing services.

use crate::{IoByteLimits, RoutedCellIo, ScheduledCellIo};
use prism_tasks::{IoReactor, NamedThreads};
use prism_world_io::{QueuedReactorFileProvider, ReactorFileConfig, ReactorFileProvider};
use std::{io, path::PathBuf, sync::Arc};

/// Manifest-checked native archive reads on host-owned IO services.
///
/// This backend never creates threads or owns the last service handle. Providers
/// and operations borrow the host's reactor/worker owners. Host teardown belongs
/// on a non-compute thread, with workers drained before the final reactor drop.
pub type ReactorCellIo<'services> =
    RoutedCellIo<ScheduledCellIo<QueuedReactorFileProvider<'services>>>;

impl<'services> ReactorCellIo<'services> {
    /// Creates a routed native backend with bounded cancellation-safe admission.
    ///
    /// `max_in_flight` must fit `native.capacity`. The provider is exclusive to
    /// this scheduler, but services can be shared with other providers. Native
    /// quotas include canceled reads through file/payload cleanup; scheduler byte
    /// limits include complete cell aggregation and undelivered results, not only
    /// native chunks. Temporary native slot/byte exhaustion waits for a wake.
    ///
    /// Register each manifest/range with [`Self::register`] before submitting it.
    /// IO jobs park while awaiting native completion; reserve enough IO lane
    /// capacity and never run consumers blocking on them on a saturated IO lane.
    /// Unsupported platforms return `io::ErrorKind::Unsupported`, with no fake
    /// blocking-read fallback. Construction does not open the file.
    pub fn reactor(
        path: impl Into<PathBuf>,
        reactor: &'services Arc<IoReactor>,
        workers: &'services NamedThreads,
        native: ReactorFileConfig,
        max_in_flight: usize,
        capacity: usize,
        limits: IoByteLimits,
    ) -> io::Result<Self> {
        if max_in_flight == 0 || max_in_flight > native.capacity || capacity == 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidInput,
                "active IO limit must be nonzero and fit native capacity; request capacity must be nonzero"));
        }
        let provider = ReactorFileProvider::new(path, reactor, workers, native)?.queued();
        let io = ScheduledCellIo::with_byte_limits(
            provider,
            max_in_flight,
            capacity,
            native.max_chunk_size,
            limits,
        )
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error.to_string()))?;
        Ok(Self::new(io))
    }
}
