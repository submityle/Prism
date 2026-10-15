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
}

struct PendingRead {
    request: CellReadRequest,
    bytes: Vec<u8>,
}

struct TrackingProvider<P> {
    provider: P,
    failed_requests: Arc<Mutex<VecDeque<IoRequestId>>>,
}

struct TrackingOperation<O> {
    request_id: IoRequestId,
    request: ReadRequest,
    operation: O,
    next_offset: u64,
    delivered: u64,
    checksum: u64,
    failed_requests: Arc<Mutex<VecDeque<IoRequestId>>>,
    failure_recorded: bool,
}

fn record_failure(
    failed_requests: &Arc<Mutex<VecDeque<IoRequestId>>>,
    request_id: IoRequestId,
    recorded: &mut bool,
) {
    if *recorded {
        return;
    }
    if let Ok(mut failed) = failed_requests.lock() {
        failed.push_back(request_id);
        *recorded = true;
    }
}

impl<O: ChunkedRead> ChunkedRead for TrackingOperation<O> {
    fn poll_chunk(
        &mut self,
        context: &mut Context<'_>,
    ) -> Poll<Result<Option<ReadChunk>, IoError>> {
        let result = self.operation.poll_chunk(context);
        match &result {
            Poll::Pending => {}
            Poll::Ready(Err(_)) => record_failure(
                &self.failed_requests,
                self.request_id,
                &mut self.failure_recorded,
            ),
            Poll::Ready(Ok(Some(chunk))) => {
                let remaining = self.request.length().saturating_sub(self.delivered);
                if chunk.offset() != self.next_offset || chunk.bytes().len() as u64 > remaining {
                    record_failure(
                        &self.failed_requests,
                        self.request_id,
                        &mut self.failure_recorded,
                    );
                } else {
                    self.next_offset += chunk.bytes().len() as u64;
                    self.delivered += chunk.bytes().len() as u64;
                    for byte in chunk.bytes() {
                        self.checksum ^= *byte as u64;
                        self.checksum = self.checksum.wrapping_mul(0x100000001b3);
                    }
                }
            }
            Poll::Ready(Ok(None)) => {
                if self.delivered != self.request.length()
                    || self
                        .request
                        .expected_checksum()
                        .is_some_and(|expected| expected != self.checksum)
                {
                    record_failure(
                        &self.failed_requests,
                        self.request_id,
                        &mut self.failure_recorded,
                    );
                }
            }
        }
        result
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
        match self.provider.start(request, request_id) {
            Ok(operation) => Ok(TrackingOperation {
                request_id,
                request: request.clone(),
                operation,
                next_offset: request.offset(),
                delivered: 0,
                checksum: prism_world_io::checksum(&[]),
                failed_requests: Arc::clone(&self.failed_requests),
                failure_recorded: false,
            }),
            Err(error) => {
                if let Ok(mut failed) = self.failed_requests.lock() {
                    failed.push_back(request_id);
                }
                Err(error)
            }
        }
    }
}

struct AdapterState<P: ReadProvider> {
    scheduler: IoScheduler<TrackingProvider<P>>,
    pending: BTreeMap<IoRequestId, PendingRead>,
    ready: VecDeque<IoAdapterEvent>,
    failed_requests: Arc<Mutex<VecDeque<IoRequestId>>>,
}

/// A std adapter that exposes [`prism_world_io::IoScheduler`] through world cell requests.
///
/// [`Self::poll`] is the asynchronous boundary and preserves chunk events. The
/// [`ChunkProvider`] implementation is a blocking compatibility bridge for providers
/// that use the scheduler's normal wake-up contract.
pub struct IoSchedulerAdapter<P: ReadProvider> {
    state: Mutex<AdapterState<P>>,
    chunk_size: usize,
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
        let failed_requests = Arc::new(Mutex::new(VecDeque::new()));
        let tracking_provider = TrackingProvider {
            provider,
            failed_requests: Arc::clone(&failed_requests),
        };
        let scheduler = IoScheduler::new(tracking_provider, max_in_flight, capacity)?;
        Ok(Self {
            state: Mutex::new(AdapterState {
                scheduler,
                pending: BTreeMap::new(),
                ready: VecDeque::new(),
                failed_requests,
            }),
            chunk_size,
        })
    }

    /// Submits a world cell request to the underlying scheduler.
    pub fn submit(&self, request: CellReadRequest) -> Result<IoRequestId, IoError> {
        let mut state = self.lock_state()?;
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
        self.state
            .lock()
            .map(|mut state| state.scheduler.cancel(request_id))
            .unwrap_or(false)
    }

    /// Returns the number of requests not yet reported to the caller.
    pub fn len(&self) -> usize {
        self.state
            .lock()
            .map(|state| state.scheduler.len())
            .unwrap_or(0)
    }

    /// Returns whether the underlying scheduler has no outstanding requests.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Polls one chunk, completion, cancellation, or provider error.
    pub fn poll(&self, context: &mut Context<'_>) -> Poll<Result<IoAdapterEvent, IoError>> {
        let mut state = match self.lock_state() {
            Ok(state) => state,
            Err(error) => return Poll::Ready(Err(error)),
        };
        if let Some(event) = state.ready.pop_front() {
            return Poll::Ready(Ok(event));
        }
        match state.scheduler.poll(context) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Err(error)) => {
                let request_id = match error {
                    IoError::Cancelled { request_id } => Some(request_id),
                    _ => state
                        .failed_requests
                        .lock()
                        .ok()
                        .and_then(|mut failed| failed.pop_front()),
                };
                if let Some(request_id) = request_id {
                    state.pending.remove(&request_id);
                }
                Poll::Ready(Err(error))
            }
            Poll::Ready(Ok(IoEvent::Chunk { request_id, chunk })) => {
                let Some(pending) = state.pending.get_mut(&request_id) else {
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
                let Some(pending) = state.pending.remove(&summary.request_id()) else {
                    return Poll::Ready(Err(internal_error("completion has no pending request")));
                };
                Poll::Ready(Ok(IoAdapterEvent::Completed {
                    request_id: summary.request_id(),
                    request: pending.request,
                    summary,
                    bytes: pending.bytes,
                }))
            }
        }
    }

    /// Reads one cell to completion using the scheduler's wake-up contract.
    pub fn read_to_completion(
        &self,
        request: &CellReadRequest,
        cancel: CancellationToken,
    ) -> Result<Vec<u8>, IoError> {
        if cancel.is_cancelled() {
            return Err(IoError::Provider {
                message: String::from("cell request was cancelled before submission"),
            });
        }
        let request_id = self.submit(*request)?;
        let parker = Arc::new(BlockingWaker::new());
        let waker = Waker::from(Arc::clone(&parker));
        let mut context = Context::from_waker(&waker);
        let mut deferred = VecDeque::new();
        loop {
            match self.poll(&mut context) {
                Poll::Pending => parker.park(),
                Poll::Ready(Err(error)) => {
                    self.requeue(deferred);
                    return Err(error);
                }
                Poll::Ready(Ok(event)) => match event {
                    IoAdapterEvent::Chunk {
                        request_id: event_id,
                        ..
                    } if event_id == request_id => {}
                    IoAdapterEvent::Completed {
                        request_id: event_id,
                        bytes,
                        ..
                    } if event_id == request_id => {
                        self.requeue(deferred);
                        return Ok(bytes);
                    }
                    other => deferred.push_back(other),
                },
            }
        }
    }

    fn lock_state(&self) -> Result<MutexGuard<'_, AdapterState<P>>, IoError> {
        self.state
            .lock()
            .map_err(|_| internal_error("adapter state poisoned"))
    }

    fn requeue(&self, mut deferred: VecDeque<IoAdapterEvent>) {
        if deferred.is_empty() {
            return;
        }
        if let Ok(mut state) = self.state.lock() {
            deferred.append(&mut state.ready);
            state.ready = deferred;
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

struct BlockingWaker {
    thread: Thread,
    notified: AtomicBool,
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
mod tests {
    use super::*;
    use crate::{ManifestHeader, SceneHost, StreamBudget, WorldFrameCoordinator};
    use alloc::vec;
    use prism_world_stream::{AabbI64, CellManifest, DemandSource, GlobalPos, SectorId};
    use std::sync::Arc;

    fn manifest(io_bytes: usize) -> CellManifest {
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

    struct MemoryOperation {
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
    struct MemoryProvider {
        bytes: Arc<Vec<u8>>,
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

    fn context() -> Context<'static> {
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
            matches!(adapter.poll(&mut context), Poll::Ready(Err(IoError::Cancelled { request_id: cancelled })) if cancelled == request_id)
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
            Poll::Ready(Err(IoError::Provider { .. }))
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
