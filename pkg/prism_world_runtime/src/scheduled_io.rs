//! Bounded asynchronous cell completions over the standard-library IO scheduler.

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::ToString;
use core::task::{Context, Poll};

use prism_world_io::{ConfigError, IoError, ReadProvider};

use crate::{
    CellIo, CellReadRequest, IoAdapterEvent, IoByteLimits, IoCompletion, IoRequestId,
    IoSchedulerAdapter, RequestId, SubmitError,
};

const CHUNK_EVENTS_PER_POLL: usize = 16;

/// Asynchronous cell IO with bounded outstanding requests and cooperative polling.
///
/// The adapter owns aggregation and validation. Chunk events are consumed and
/// dropped immediately; only terminal events are exposed through [`CellIo`].
/// A poll processes at most sixteen chunk events before waking its caller and
/// yielding. Provider callbacks must themselves obey a bounded polling contract;
/// use [`prism_world_io::ThreadedFileProvider`] for worker-backed filesystem IO.
///
/// Requests retain their capacity and declared-byte charge until their terminal
/// completion is polled, including after cancellation. The ID registry retains
/// only the scheduler identity needed to cancel each outstanding request.
pub struct ScheduledCellIo<P: ReadProvider> {
    adapter: IoSchedulerAdapter<P>,
    requests: BTreeMap<RequestId, IoRequestId>,
}

impl<P: ReadProvider> ScheduledCellIo<P> {
    /// Creates a backend with explicit active, outstanding, and chunk-size bounds.
    ///
    /// This constructor does not impose a declared-byte limit. Use
    /// [`Self::with_byte_limits`] to bound aggregate cell storage as well.
    pub fn new(
        provider: P,
        max_in_flight: usize,
        capacity: usize,
        chunk_size: usize,
    ) -> Result<Self, ConfigError> {
        Ok(Self {
            adapter: IoSchedulerAdapter::with_chunk_size(
                provider,
                max_in_flight,
                capacity,
                chunk_size,
            )?,
            requests: BTreeMap::new(),
        })
    }

    /// Creates a backend with explicit request-count, chunk-size, and byte limits.
    ///
    /// Limits apply to the submitted request's declared bytes, including its cell
    /// header. See [`IoByteLimits`] for the distinction from allocator/RSS bounds.
    pub fn with_byte_limits(
        provider: P,
        max_in_flight: usize,
        capacity: usize,
        chunk_size: usize,
        limits: IoByteLimits,
    ) -> Result<Self, ConfigError> {
        Ok(Self {
            adapter: IoSchedulerAdapter::with_byte_limits(
                provider,
                max_in_flight,
                capacity,
                chunk_size,
                limits,
            )?,
            requests: BTreeMap::new(),
        })
    }

    /// Submits a cell request, retaining its resolved range, checksum, and priority.
    ///
    /// File routing can resolve a request before calling this method. Rejected
    /// submissions consume neither a registry slot nor a scheduler identity.
    pub fn submit(&mut self, request: CellReadRequest) -> Result<RequestId, SubmitError> {
        let scheduler_id = self.adapter.submit(request).map_err(submit_error)?;
        let request_id = RequestId(scheduler_id.get());
        self.requests.insert(request_id, scheduler_id);
        Ok(request_id)
    }

    /// Cancels a queued or active request using its retained scheduler identity.
    ///
    /// Cancellation is reported by a subsequent completion poll and does not
    /// release admission immediately. Unknown, retired, or already cancelled
    /// identities return `false`.
    pub fn cancel(&mut self, request_id: RequestId) -> bool {
        self.requests
            .get(&request_id)
            .is_some_and(|&scheduler_id| self.adapter.cancel(scheduler_id))
    }

    /// Returns requests whose terminal completion has not yet been consumed.
    #[must_use]
    pub fn in_flight(&self) -> usize {
        self.requests.len()
    }

    /// Polls one cell completion, yielding cooperatively after sixteen chunks.
    ///
    /// Request failures become [`IoCompletion::Failed`], cancellations become
    /// [`IoCompletion::Cancelled`], and internal adapter errors propagate as
    /// `Poll::Ready(Err(_))`. This method never waits for a request to finish.
    pub fn poll_completion(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Result<IoCompletion<IoError>, IoError>> {
        for _ in 0..CHUNK_EVENTS_PER_POLL {
            let event = match self.adapter.poll(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Err(error)) => return Poll::Ready(Err(error)),
                Poll::Ready(Ok(event)) => event,
            };
            let scheduler_id = event.request_id();
            let request_id = RequestId(scheduler_id.get());
            if self.requests.get(&request_id) != Some(&scheduler_id) {
                return Poll::Ready(Err(IoError::Provider {
                    message: format!(
                        "scheduled cell event has no registered request: {}",
                        scheduler_id.get()
                    ),
                }));
            }
            let completion = match event {
                // The adapter already aggregated these bytes. Dropping the
                // event here avoids retaining a second queue of cell payloads.
                IoAdapterEvent::Chunk { .. } => continue,
                IoAdapterEvent::Completed { request, bytes, .. } => IoCompletion::Completed {
                    request_id,
                    request,
                    bytes,
                },
                IoAdapterEvent::Failed {
                    request,
                    error: IoError::Cancelled { .. },
                    ..
                } => IoCompletion::Cancelled {
                    request_id,
                    request,
                },
                IoAdapterEvent::Failed { request, error, .. } => IoCompletion::Failed {
                    request_id,
                    request,
                    error,
                },
            };
            self.requests.remove(&request_id);
            return Poll::Ready(Ok(completion));
        }
        // The adapter unregisters the caller after each ready event. Explicitly
        // schedule another turn so a continuously ready provider cannot stall.
        cx.waker().wake_by_ref();
        Poll::Pending
    }
}

impl<P: ReadProvider> CellIo for ScheduledCellIo<P> {
    type Error = IoError;

    fn submit(&mut self, request: CellReadRequest) -> Result<RequestId, SubmitError> {
        ScheduledCellIo::submit(self, request)
    }

    fn cancel(&mut self, id: RequestId) -> bool {
        ScheduledCellIo::cancel(self, id)
    }

    fn in_flight(&self) -> usize {
        ScheduledCellIo::in_flight(self)
    }

    fn poll_completion(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Result<IoCompletion<Self::Error>, Self::Error>> {
        ScheduledCellIo::poll_completion(self, cx)
    }
}

#[cfg(feature = "tasks")]
impl<P: ReadProvider> crate::PriorityCellIo for ScheduledCellIo<P> {
    fn submit_with_priority(
        &mut self,
        request: CellReadRequest,
        priority: alloc::sync::Arc<prism_tasks::PriorityCell>,
    ) -> Result<RequestId, SubmitError> {
        let scheduler_id = self
            .adapter
            .submit_with_priority(request, priority)
            .map_err(submit_error)?;
        let request_id = RequestId(scheduler_id.get());
        self.requests.insert(request_id, scheduler_id);
        Ok(request_id)
    }
}

fn submit_error(error: IoError) -> SubmitError {
    match error {
        IoError::Backpressure { capacity } => SubmitError::Backpressure { capacity },
        IoError::RequestIdExhausted => SubmitError::RequestIdExhausted,
        IoError::Provider { message } => SubmitError::Provider { message },
        error => SubmitError::Provider {
            message: error.to_string(),
        },
    }
}
