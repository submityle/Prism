//! Standard-library bridge from world cell requests to the chunked IO scheduler.

use alloc::collections::{BTreeMap, VecDeque};
use alloc::string::String;
use alloc::vec::Vec;
use core::task::{Context, Poll, Waker};

use prism_world_io::{
    ChunkedRead, ConfigError, IoError, IoEvent, IoScheduler, ReadChunk, ReadProvider, ReadRequest,
    ReadSummary,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::Wake;
use std::thread::{self, Thread};

use crate::{CancellationToken, CellReadRequest, ChunkProvider};

/// The request identity assigned by [`prism_world_io::IoScheduler`].
pub type IoRequestId = prism_world_io::RequestId;

/// One event emitted by [`IoSchedulerAdapter::poll`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum IoAdapterEvent {
    /// A chunk returned by the underlying scheduler.
    Chunk {
        /// Scheduler request identity.
        request_id: IoRequestId,
        /// World request metadata.
        request: CellReadRequest,
        /// Returned chunk.
        chunk: ReadChunk,
    },
    /// All chunks for a request have been collected and validated.
    Completed {
        /// Scheduler request identity.
        request_id: IoRequestId,
        /// World request metadata.
        request: CellReadRequest,
        /// Scheduler completion accounting.
        summary: ReadSummary,
        /// Concatenated bytes returned by the provider.
        bytes: Vec<u8>,
    },
    /// A provider or scheduler validation failure associated with one request.
    Failed {
        /// Scheduler request identity.
        request_id: IoRequestId,
        /// World request metadata.
        request: CellReadRequest,
        /// Failure returned by the scheduler.
        error: IoError,
    },
}

struct PendingRead {
    request: CellReadRequest,
    bytes: Vec<u8>,
}

struct TrackingProvider<P> {
    provider: P,
    callback_request: Arc<Mutex<Option<IoRequestId>>>,
}

struct TrackingOperation<O> {
    request_id: IoRequestId,
    operation: O,
    callback_request: Arc<Mutex<Option<IoRequestId>>>,
}

impl<O: ChunkedRead> ChunkedRead for TrackingOperation<O> {
    fn poll_chunk(
        &mut self,
        context: &mut Context<'_>,
    ) -> Poll<Result<Option<ReadChunk>, IoError>> {
        match self.callback_request.lock() {
            Ok(mut current) => *current = Some(self.request_id),
            Err(_) => return Poll::Ready(Err(internal_error("request tracking poisoned"))),
        }
        self.operation
            .poll_chunk(context)
            .map(|result| result.map_err(|error| normalize_cancellation(error, self.request_id)))
    }

    fn cancel(&mut self) {
        self.operation.cancel();
    }
}

impl<P: ReadProvider> ReadProvider for TrackingProvider<P> {
    type Operation = TrackingOperation<P::Operation>;

    fn start(
        &self,
        request: &ReadRequest,
        request_id: IoRequestId,
    ) -> Result<Self::Operation, IoError> {
        *self
            .callback_request
            .lock()
            .map_err(|_| internal_error("request tracking poisoned"))? = Some(request_id);
        self.provider
            .start(request, request_id)
            .map_err(|error| normalize_cancellation(error, request_id))
            .map(|operation| TrackingOperation {
                request_id,
                operation,
                callback_request: Arc::clone(&self.callback_request),
            })
    }
}

struct AdapterState<P: ReadProvider> {
    scheduler: IoScheduler<TrackingProvider<P>>,
    pending: BTreeMap<IoRequestId, PendingRead>,
    ready: VecDeque<IoAdapterEvent>,
    callback_request: Arc<Mutex<Option<IoRequestId>>>,
    blocking: BTreeMap<IoRequestId, Waker>,
}

/// A std adapter that exposes [`prism_world_io::IoScheduler`] through world cell requests.
///
/// [`Self::poll`] is the asynchronous boundary and preserves chunk events. The
/// [`ChunkProvider`] implementation is a blocking compatibility bridge for providers
/// that use the scheduler's normal wake-up contract.
pub struct IoSchedulerAdapter<P: ReadProvider> {
    state: Mutex<AdapterState<P>>,
    chunk_size: usize,
    capacity: usize,
    wake: Arc<SchedulerWake>,
}

impl<P: ReadProvider> IoSchedulerAdapter<P> {
    /// Creates an adapter using the default 64 KiB scheduler chunk size.
    pub fn new(provider: P, max_in_flight: usize, capacity: usize) -> Result<Self, ConfigError> {
        Self::with_chunk_size(provider, max_in_flight, capacity, 64 * 1024)
    }

    /// Creates an adapter with an explicit scheduler chunk size.
    pub fn with_chunk_size(
        provider: P,
        max_in_flight: usize,
        capacity: usize,
        chunk_size: usize,
    ) -> Result<Self, ConfigError> {
        if chunk_size == 0 {
            return Err(ConfigError::ZeroChunkSize);
        }
        let callback_request = Arc::new(Mutex::new(None));
        let tracking_provider = TrackingProvider {
            provider,
            callback_request: Arc::clone(&callback_request),
        };
        let scheduler = IoScheduler::new(tracking_provider, max_in_flight, capacity)?;
        Ok(Self {
            state: Mutex::new(AdapterState {
                scheduler,
                pending: BTreeMap::new(),
                ready: VecDeque::new(),
                callback_request,
                blocking: BTreeMap::new(),
            }),
            chunk_size,
            capacity,
            wake: Arc::new(SchedulerWake::default()),
        })
    }

    /// Submits a world cell request to the underlying scheduler.
    pub fn submit(&self, request: CellReadRequest) -> Result<IoRequestId, IoError> {
        let mut state = self.lock_state()?;
        let id = self.submit_locked(&mut state, request)?;
        drop(state);
        self.wake.wake_by_ref();
        Ok(id)
    }

    fn submit_locked(
        &self,
        state: &mut AdapterState<P>,
        request: CellReadRequest,
    ) -> Result<IoRequestId, IoError> {
        if state.pending.len() >= self.capacity {
            return Err(IoError::Backpressure {
                capacity: self.capacity,
            });
        }
        let io_request = request
            .io_request()
            .with_chunk_size(self.chunk_size)
            .map_err(|_| IoError::Provider {
                message: String::from("adapter chunk size is invalid"),
            })?;
        let request_id = state.scheduler.submit(io_request)?;
        state.pending.insert(
            request_id,
            PendingRead {
                request,
                bytes: Vec::new(),
            },
        );
        Ok(request_id)
    }

    /// Cancels a queued or active scheduler request.
    pub fn cancel(&self, request_id: IoRequestId) -> bool {
        let cancelled = self
            .state
            .lock()
            .map(|mut state| state.scheduler.cancel(request_id))
            .unwrap_or(false);
        if cancelled {
            self.wake.wake_by_ref();
        }
        cancelled
    }

    /// Returns the number of requests not yet reported to the caller.
    pub fn len(&self) -> usize {
        self.state
            .lock()
            .map(|state| state.pending.len())
            .unwrap_or(0)
    }

    /// Returns whether the underlying scheduler has no outstanding requests.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Polls one chunk, completion, or request-associated failure (including cancellation).
    ///
    /// Request failures are [`IoAdapterEvent::Failed`]; `Err` is reserved for
    /// an internal adapter failure. Events reserved by blocking readers are not
    /// exposed here. All waiting readers are woken when any provider makes progress.
    pub fn poll(&self, context: &mut Context<'_>) -> Poll<Result<IoAdapterEvent, IoError>> {
        self.poll_for(context, None)
    }

    fn poll_for(
        &self,
        context: &mut Context<'_>,
        target: Option<IoRequestId>,
    ) -> Poll<Result<IoAdapterEvent, IoError>> {
        self.wake.register(context.waker());
        let mut state = match self.lock_state() {
            Ok(state) => state,
            Err(error) => {
                self.wake.unregister(context.waker());
                return Poll::Ready(Err(error));
            }
        };
        let accepts = |event: &IoAdapterEvent, state: &AdapterState<P>| match target {
            Some(id) => event.request_id() == id,
            None => !state.blocking.contains_key(&event.request_id()),
        };
        if let Some(index) = state.ready.iter().position(|event| accepts(event, &state)) {
            let event = state.ready.remove(index).expect("ready event index");
            state.retire(&event);
            self.wake.unregister(context.waker());
            return Poll::Ready(Ok(event));
        }
        let scheduler_waker = Waker::from(Arc::clone(&self.wake));
        let mut scheduler_context = Context::from_waker(&scheduler_waker);
        let result = state.poll_scheduler(&mut scheduler_context);
        match result {
            Poll::Ready(Ok(event)) if !accepts(&event, &state) => {
                let waiter = state.blocking.get(&event.request_id()).cloned();
                state.ready.push_back(event);
                drop(state);
                if let Some(waiter) = waiter {
                    waiter.wake();
                }
                // Progress was made even if this event belongs to another reader.
                self.wake.wake_by_ref();
                Poll::Pending
            }
            Poll::Ready(Ok(event)) => {
                state.retire(&event);
                self.wake.unregister(context.waker());
                Poll::Ready(Ok(event))
            }
            Poll::Ready(Err(error)) => {
                self.wake.unregister(context.waker());
                Poll::Ready(Err(error))
            }
            Poll::Pending => Poll::Pending,
        }
    }

    /// Reads one cell to completion using the scheduler's wake-up contract.
    ///
    /// Other requests remain observable through [`Self::poll`], including failures.
    /// Blocking readers reserve their own events so concurrent readers cannot
    /// consume one another's results. The file provider performs bounded blocking
    /// reads; this method does not turn filesystem calls into worker-thread IO.
    pub fn read_to_completion(
        &self,
        request: &CellReadRequest,
        cancel: CancellationToken,
    ) -> Result<Vec<u8>, IoError> {
        if cancel.is_cancelled() {
            return Err(internal_error(
                "cell request was cancelled before submission",
            ));
        }
        let parker = Arc::new(BlockingWaker::new());
        let waker = Waker::from(Arc::clone(&parker));
        let request_id = {
            let mut state = self.lock_state()?;
            let id = self.submit_locked(&mut state, *request)?;
            state.blocking.insert(id, waker.clone());
            id
        };
        let mut context = Context::from_waker(&waker);
        loop {
            match self.poll_for(&mut context, Some(request_id)) {
                Poll::Pending => parker.park(),
                Poll::Ready(Err(error)) => return Err(error),
                Poll::Ready(Ok(IoAdapterEvent::Chunk { .. })) => {}
                Poll::Ready(Ok(IoAdapterEvent::Completed { bytes, .. })) => return Ok(bytes),
                Poll::Ready(Ok(IoAdapterEvent::Failed { error, .. })) => return Err(error),
            }
        }
    }

    fn lock_state(&self) -> Result<MutexGuard<'_, AdapterState<P>>, IoError> {
        self.state
            .lock()
            .map_err(|_| internal_error("adapter state poisoned"))
    }
}

impl IoAdapterEvent {
    /// Returns the identity of the request that produced this event.
    #[must_use]
    pub const fn request_id(&self) -> IoRequestId {
        match self {
            Self::Chunk { request_id, .. }
            | Self::Completed { request_id, .. }
            | Self::Failed { request_id, .. } => *request_id,
        }
    }
}

impl<P: ReadProvider> AdapterState<P> {
    fn retire(&mut self, event: &IoAdapterEvent) {
        if matches!(
            event,
            IoAdapterEvent::Completed { .. } | IoAdapterEvent::Failed { .. }
        ) {
            self.pending.remove(&event.request_id());
            self.blocking.remove(&event.request_id());
        }
    }

    fn poll_scheduler(
        &mut self,
        context: &mut Context<'_>,
    ) -> Poll<Result<IoAdapterEvent, IoError>> {
        // The scheduler is serialized under the adapter lock. Every provider
        // callback records its identity, so a subsequent validation failure is
        // associated with that exact callback, not a stale failure FIFO.
        match self.callback_request.lock() {
            Ok(mut current) => *current = None,
            Err(_) => return Poll::Ready(Err(internal_error("request tracking poisoned"))),
        }
        match self.scheduler.poll(context) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Err(error)) => {
                let callback = match self.callback_request.lock() {
                    Ok(mut current) => current.take(),
                    Err(_) => return Poll::Ready(Err(internal_error("request tracking poisoned"))),
                };
                // Scheduler cancellation can bypass operation polling after it
                // starts a different queued request. Its explicit ID wins over
                // the last provider callback; provider cancellations have been
                // normalized by the tracking wrapper above.
                let request_id = match error {
                    IoError::Cancelled { request_id } => Some(request_id),
                    _ => callback,
                };
                let Some(request_id) = request_id else {
                    return Poll::Ready(Err(error));
                };
                let Some(pending) = self.pending.get_mut(&request_id) else {
                    return Poll::Ready(Err(internal_error("failure has no pending request")));
                };
                pending.bytes.clear();
                let error = match error {
                    IoError::Cancelled { .. } => IoError::Cancelled { request_id },
                    other => other,
                };
                Poll::Ready(Ok(IoAdapterEvent::Failed {
                    request_id,
                    request: pending.request,
                    error,
                }))
            }
            Poll::Ready(Ok(IoEvent::Chunk { request_id, chunk })) => {
                let Some(pending) = self.pending.get_mut(&request_id) else {
                    return Poll::Ready(Err(internal_error("chunk has no pending request")));
                };
                pending.bytes.extend_from_slice(chunk.bytes());
                let request = pending.request;
                Poll::Ready(Ok(IoAdapterEvent::Chunk {
                    request_id,
                    request,
                    chunk,
                }))
            }
            Poll::Ready(Ok(IoEvent::Completed(summary))) => {
                let Some(pending) = self.pending.get_mut(&summary.request_id()) else {
                    return Poll::Ready(Err(internal_error("completion has no pending request")));
                };
                Poll::Ready(Ok(IoAdapterEvent::Completed {
                    request_id: summary.request_id(),
                    request: pending.request,
                    summary,
                    bytes: core::mem::take(&mut pending.bytes),
                }))
            }
        }
    }
}

impl<P: ReadProvider> ChunkProvider for IoSchedulerAdapter<P> {
    type Error = IoError;

    fn read_chunk(
        &self,
        request: &CellReadRequest,
        cancel: CancellationToken,
    ) -> Result<Vec<u8>, Self::Error> {
        self.read_to_completion(request, cancel)
    }
}

fn internal_error(message: &str) -> IoError {
    IoError::Provider {
        message: String::from(message),
    }
}

fn normalize_cancellation(error: IoError, request_id: IoRequestId) -> IoError {
    match error {
        IoError::Cancelled { .. } => IoError::Cancelled { request_id },
        other => other,
    }
}

struct BlockingWaker {
    thread: Thread,
    notified: AtomicBool,
}

#[derive(Default)]
struct SchedulerWake {
    waiters: Mutex<Vec<Waker>>,
}

impl SchedulerWake {
    fn register(&self, waker: &Waker) {
        let mut waiters = self
            .waiters
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if !waiters.iter().any(|waiting| waiting.will_wake(waker)) {
            waiters.push(waker.clone());
        }
    }

    fn unregister(&self, waker: &Waker) {
        self.waiters
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .retain(|waiting| !waiting.will_wake(waker));
    }
}

impl Wake for SchedulerWake {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        let waiters = core::mem::take(
            &mut *self
                .waiters
                .lock()
                .unwrap_or_else(|error| error.into_inner()),
        );
        for waiter in waiters {
            waiter.wake();
        }
    }
}

impl BlockingWaker {
    fn new() -> Self {
        Self {
            thread: thread::current(),
            notified: AtomicBool::new(false),
        }
    }

    fn park(&self) {
        while !self.notified.swap(false, Ordering::Acquire) {
            thread::park();
        }
    }
}

impl Wake for BlockingWaker {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.notified.store(true, Ordering::Release);
        self.thread.unpark();
    }
}

#[cfg(test)]
#[path = "io_adapter_tests.rs"]
mod correlation_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ManifestHeader, SceneHost, StreamBudget, WorldFrameCoordinator};
    use alloc::vec;
    use prism_world_stream::{AabbI64, CellManifest, DemandSource, GlobalPos, SectorId};
    use std::sync::Arc;

    pub(super) fn manifest(io_bytes: usize) -> CellManifest {
        CellManifest {
            sector: SectorId::new(0, 0, 0, 0),
            bounds: AabbI64::new(GlobalPos::new(0, 0, 0), GlobalPos::new(1, 1, 1))
                .expect("valid bounds"),
            scene_key: 7,
            content_version: 1,
            derived_hash: 9,
            io_bytes: io_bytes as u64,
        }
    }

    pub struct MemoryOperation {
        request_id: IoRequestId,
        bytes: Vec<u8>,
        offset: usize,
        chunk_size: usize,
        cancelled: bool,
    }

    impl ChunkedRead for MemoryOperation {
        fn poll_chunk(
            &mut self,
            _context: &mut Context<'_>,
        ) -> Poll<Result<Option<ReadChunk>, IoError>> {
            if self.cancelled {
                return Poll::Ready(Err(IoError::Cancelled {
                    request_id: self.request_id,
                }));
            }
            if self.offset == self.bytes.len() {
                return Poll::Ready(Ok(None));
            }
            let end = (self.offset + self.chunk_size).min(self.bytes.len());
            let chunk = ReadChunk::new(self.offset as u64, self.bytes[self.offset..end].to_vec());
            self.offset = end;
            Poll::Ready(Ok(Some(chunk)))
        }

        fn cancel(&mut self) {
            self.cancelled = true;
        }
    }

    #[derive(Clone)]
    pub(super) struct MemoryProvider {
        pub(super) bytes: Arc<Vec<u8>>,
    }

    impl ReadProvider for MemoryProvider {
        type Operation = MemoryOperation;

        fn start(
            &self,
            request: &ReadRequest,
            request_id: IoRequestId,
        ) -> Result<Self::Operation, IoError> {
            Ok(MemoryOperation {
                request_id,
                bytes: self.bytes.as_ref().clone(),
                offset: request.offset() as usize,
                chunk_size: request.chunk_size(),
                cancelled: false,
            })
        }
    }

    struct FailingProvider;

    impl ReadProvider for FailingProvider {
        type Operation = MemoryOperation;

        fn start(
            &self,
            _request: &ReadRequest,
            _request_id: IoRequestId,
        ) -> Result<Self::Operation, IoError> {
            Err(IoError::Provider {
                message: String::from("memory provider failure"),
            })
        }
    }

    pub(super) fn context() -> Context<'static> {
        Context::from_waker(Waker::noop())
    }

    #[test]
    fn memory_provider_emits_chunks_and_completed_bytes() {
        let payload = vec![1, 2, 3, 4, 5, 6, 7];
        let adapter = IoSchedulerAdapter::with_chunk_size(
            MemoryProvider {
                bytes: Arc::new(payload.clone()),
            },
            1,
            2,
            3,
        )
        .expect("adapter config");
        let request = CellReadRequest::new(manifest(payload.len()), 11);
        let request_id = adapter.submit(request).expect("submit");
        let mut context = context();
        let mut chunks = 0;
        loop {
            match adapter.poll(&mut context) {
                Poll::Ready(Ok(IoAdapterEvent::Chunk {
                    request_id: event_id,
                    chunk,
                    ..
                })) => {
                    assert_eq!(event_id, request_id);
                    assert!(!chunk.bytes().is_empty());
                    chunks += 1;
                }
                Poll::Ready(Ok(IoAdapterEvent::Completed {
                    request_id: event_id,
                    bytes,
                    summary,
                    ..
                })) => {
                    assert_eq!(event_id, request_id);
                    assert_eq!(bytes, payload);
                    assert_eq!(summary.length(), 7);
                    break;
                }
                Poll::Ready(Err(error)) => panic!("unexpected IO error: {error}"),
                Poll::Ready(Ok(IoAdapterEvent::Failed { error, .. })) => {
                    panic!("unexpected request failure: {error}")
                }
                Poll::Pending => panic!("memory provider should be immediately ready"),
            }
        }
        assert_eq!(chunks, 3);
        assert!(adapter.is_empty());
    }

    #[test]
    fn cancellation_and_backpressure_are_forwarded() {
        let payload = vec![0; 4];
        let adapter = IoSchedulerAdapter::new(
            MemoryProvider {
                bytes: Arc::new(payload.clone()),
            },
            1,
            1,
        )
        .expect("adapter config");
        let request = CellReadRequest::new(manifest(payload.len()), 1);
        let request_id = adapter.submit(request).expect("submit");
        assert!(matches!(
            adapter.submit(request),
            Err(IoError::Backpressure { capacity: 1 })
        ));
        assert!(adapter.cancel(request_id));
        let mut context = context();
        assert!(
            matches!(adapter.poll(&mut context), Poll::Ready(Ok(IoAdapterEvent::Failed { request_id: cancelled, error: IoError::Cancelled { .. }, .. })) if cancelled == request_id)
        );
        assert!(adapter.is_empty());
    }

    #[test]
    fn provider_errors_are_reported_and_released() {
        let adapter = IoSchedulerAdapter::new(FailingProvider, 1, 1).expect("adapter config");
        let request = CellReadRequest::new(manifest(4), 1);
        let request_id = adapter.submit(request).expect("submit");
        let mut context = context();
        assert!(matches!(
            adapter.poll(&mut context),
            Poll::Ready(Ok(IoAdapterEvent::Failed { request_id: failed, error: IoError::Provider { .. }, .. })) if failed == request_id
        ));
        assert_eq!(
            adapter.len(),
            0,
            "failed request {request_id} must be released"
        );
    }

    #[derive(Default)]
    struct Host {
        payloads: Vec<Vec<u8>>,
    }

    impl SceneHost for Host {
        type Error = ();

        fn load_cell(
            &mut self,
            _manifest: &CellManifest,
            payload: &[u8],
        ) -> Result<(), Self::Error> {
            self.payloads.push(payload.to_vec());
            Ok(())
        }

        fn evict_cell(&mut self, _sector: SectorId) -> Result<(), Self::Error> {
            Ok(())
        }
    }

    #[test]
    fn memory_scheduler_adapter_loads_through_world_coordinator() {
        let payload = b"scene-payload".to_vec();
        let mut manifest = manifest(payload.len() + crate::CELL_HEADER_SIZE);
        manifest.io_bytes = (payload.len() + crate::CELL_HEADER_SIZE) as u64;
        let mut bytes = ManifestHeader::from_manifest(manifest).encode();
        bytes.extend_from_slice(&payload);
        let adapter = IoSchedulerAdapter::with_chunk_size(
            MemoryProvider {
                bytes: Arc::new(bytes),
            },
            1,
            1,
            5,
        )
        .expect("adapter config");
        let mut coordinator = WorldFrameCoordinator::new(adapter, 1).expect("runtime config");
        let mut host = Host::default();
        let demand = DemandSource::new(manifest.sector, 0, 1, 10).expect("demand");
        let report = coordinator.plan_frame(
            &[demand],
            &[manifest],
            &[],
            StreamBudget {
                io_bytes: manifest.io_bytes,
                max_actions: 1,
            },
            &mut host,
        );
        assert_eq!(report.submitted, 1);
        assert!(coordinator.poll_io(&mut host));
        assert_eq!(host.payloads, vec![payload]);
    }
}
