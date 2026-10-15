//! Standard-library bridge from world cell requests to the chunked IO scheduler.

use alloc::collections::{BTreeMap, BTreeSet, VecDeque};
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::ops::{Deref, DerefMut};
use core::task::{Context, Poll, Waker};

use prism_world_io::{
    ChunkedRead, ConfigError, IoError, IoEvent, IoScheduler, ReadChunk, ReadProvider, ReadRequest,
    ReadSummary,
};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::Wake;
use std::thread::{self, Thread};

use crate::{CancellationToken, CellReadRequest, ChunkProvider};

/// The request identity assigned by [`prism_world_io::IoScheduler`].
pub type IoRequestId = prism_world_io::RequestId;

/// Admission limits for declared cell bytes, including the fixed manifest header.
///
/// These are logical byte limits, not an allocator/RSS limit: buffered chunk
/// events can retain another copy of the aggregate payload. Zero is allowed
/// and admits only empty requests; values are never silently clamped.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IoByteLimits {
    /// Maximum declared bytes for one request.
    pub max_request_bytes: u64,
    /// Maximum sum of declared bytes for all requests whose terminal event
    /// has not been consumed, including queued, active, and buffered results.
    pub max_total_bytes: u64,
}

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
    allocation_error: Option<IoError>,
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
    blocking: BTreeSet<IoRequestId>,
    admitted_bytes: u64,
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
    byte_limits: Option<IoByteLimits>,
    wake: Arc<SchedulerWake>,
}

// Fields drop in declaration order: unlock adapter state before dispatching any
// wake deferred by a provider callback (including on unwinding/early returns).
struct AdapterGuard<'a, P: ReadProvider> {
    state: MutexGuard<'a, AdapterState<P>>,
    _busy: WakeBusy<'a>,
}

impl<P: ReadProvider> Deref for AdapterGuard<'_, P> {
    type Target = AdapterState<P>;

    fn deref(&self) -> &Self::Target {
        &self.state
    }
}

impl<P: ReadProvider> DerefMut for AdapterGuard<'_, P> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.state
    }
}

impl<P: ReadProvider> IoSchedulerAdapter<P> {
    /// Creates an adapter using the default 64 KiB scheduler chunk size.
    pub fn new(provider: P, max_in_flight: usize, capacity: usize) -> Result<Self, ConfigError> {
        Self::with_chunk_size(provider, max_in_flight, capacity, 64 * 1024)
    }

    /// Creates an adapter with an explicit scheduler chunk size.
    ///
    /// This compatibility constructor has no configured byte limits. Accounting
    /// overflow is still rejected; use [`Self::with_byte_limits`] for byte admission.
    pub fn with_chunk_size(
        provider: P,
        max_in_flight: usize,
        capacity: usize,
        chunk_size: usize,
    ) -> Result<Self, ConfigError> {
        Self::configured(provider, max_in_flight, capacity, chunk_size, None)
    }

    /// Creates an adapter with explicit request-count and declared-byte limits.
    ///
    /// Admission rejects a request exceeding either byte limit before assigning
    /// an ID or starting IO. Limits are independent: a per-request limit larger
    /// than the total limit is preserved, not clamped. Terminal results hold
    /// their declared bytes until consumed. See [`IoByteLimits`] for memory scope.
    pub fn with_byte_limits(
        provider: P,
        max_in_flight: usize,
        capacity: usize,
        chunk_size: usize,
        limits: IoByteLimits,
    ) -> Result<Self, ConfigError> {
        Self::configured(provider, max_in_flight, capacity, chunk_size, Some(limits))
    }

    fn configured(
        provider: P,
        max_in_flight: usize,
        capacity: usize,
        chunk_size: usize,
        byte_limits: Option<IoByteLimits>,
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
                blocking: BTreeSet::new(),
                admitted_bytes: 0,
            }),
            chunk_size,
            capacity,
            byte_limits,
            wake: Arc::new(SchedulerWake::default()),
        })
    }

    /// Submits a world cell request to the underlying scheduler.
    ///
    /// A byte-admission rejection is an [`IoError::Provider`] with explicit
    /// requested/limit values; it has no request ID and consumes no quota.
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
        let declared = request.manifest().io_bytes;
        if let Some(limits) = self.byte_limits
            && declared > limits.max_request_bytes
        {
            return Err(IoError::Provider {
                message: format!(
                    "request declares {declared} bytes, exceeding per-request limit {}",
                    limits.max_request_bytes
                ),
            });
        }
        let admitted_bytes =
            state
                .admitted_bytes
                .checked_add(declared)
                .ok_or_else(|| IoError::Provider {
                    message: format!(
                        "declared byte admission overflow: {} + {declared}",
                        state.admitted_bytes
                    ),
                })?;
        if let Some(limits) = self.byte_limits
            && admitted_bytes > limits.max_total_bytes
        {
            return Err(IoError::Provider {
                message: format!(
                    "total declared bytes {admitted_bytes} exceed total limit {}",
                    limits.max_total_bytes
                ),
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
                allocation_error: None,
            },
        );
        state.admitted_bytes = admitted_bytes;
        Ok(request_id)
    }

    /// Returns the configured byte limits, or `None` for compatibility constructors.
    #[must_use]
    pub const fn byte_limits(&self) -> Option<IoByteLimits> {
        self.byte_limits
    }

    /// Returns declared bytes held by all requests with unconsumed terminal events.
    ///
    /// Unlike allocator capacity, this does not decrease when an operation fails
    /// or when individual chunk events are consumed.
    pub fn admitted_bytes(&self) -> Result<u64, IoError> {
        Ok(self.lock_state()?.admitted_bytes)
    }

    /// Cancels a queued or active scheduler request.
    pub fn cancel(&self, request_id: IoRequestId) -> bool {
        let cancelled = self
            .lock_state()
            .map(|mut state| state.scheduler.cancel(request_id))
            .unwrap_or(false);
        if cancelled {
            self.wake.wake_by_ref();
        }
        cancelled
    }

    /// Returns the number of requests not yet reported to the caller.
    pub fn len(&self) -> usize {
        self.lock_state()
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
    ///
    /// Public polling supports one asynchronous consumer. Each call replaces
    /// that consumer's registered Waker with the supplied Context's Waker;
    /// after `Pending`, only the latest registration receives the next wake.
    /// Independent tasks must not rely on separate public poll registrations.
    /// Blocking readers retain separate registrations for their request IDs.
    pub fn poll(&self, context: &mut Context<'_>) -> Poll<Result<IoAdapterEvent, IoError>> {
        self.poll_for(context, None)
    }

    fn poll_for(
        &self,
        context: &mut Context<'_>,
        target: Option<IoRequestId>,
    ) -> Poll<Result<IoAdapterEvent, IoError>> {
        let registration = self.wake.register(target, context.waker());
        let scheduler_waker = Waker::from(Arc::clone(&self.wake));
        let mut scheduler_context = Context::from_waker(&scheduler_waker);
        let mut state = match self.lock_state() {
            Ok(state) => state,
            Err(error) => {
                self.wake.unregister(&registration);
                return Poll::Ready(Err(error));
            }
        };
        let accepts = |event: &IoAdapterEvent, state: &AdapterState<P>| match target {
            Some(id) => event.request_id() == id,
            None => !state.blocking.contains(&event.request_id()),
        };
        if let Some(index) = state.ready.iter().position(|event| accepts(event, &state)) {
            let event = state.ready.remove(index).expect("ready event index");
            state.retire(&event);
            drop(state);
            self.wake.unregister(&registration);
            return Poll::Ready(Ok(event));
        }
        let result = state.poll_scheduler(&mut scheduler_context);
        match result {
            Poll::Ready(Ok(event)) if !accepts(&event, &state) => {
                state.ready.push_back(event);
                drop(state);
                // Progress was made even if this event belongs to another reader.
                self.wake.wake_by_ref();
                Poll::Pending
            }
            Poll::Ready(Ok(event)) => {
                state.retire(&event);
                drop(state);
                self.wake.unregister(&registration);
                Poll::Ready(Ok(event))
            }
            Poll::Ready(Err(error)) => {
                drop(state);
                self.wake.unregister(&registration);
                Poll::Ready(Err(error))
            }
            Poll::Pending => {
                drop(state);
                Poll::Pending
            }
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
            state.blocking.insert(id);
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

    fn lock_state(&self) -> Result<AdapterGuard<'_, P>, IoError> {
        let busy = self.wake.begin_busy();
        let state = self
            .state
            .lock()
            .map_err(|_| internal_error("adapter state poisoned"))?;
        Ok(AdapterGuard { state, _busy: busy })
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
            self.retire_request(event.request_id());
        }
    }

    fn retire_request(&mut self, request_id: IoRequestId) {
        if let Some(pending) = self.pending.remove(&request_id) {
            self.admitted_bytes = self
                .admitted_bytes
                .saturating_sub(pending.request.manifest().io_bytes);
        }
        self.blocking.remove(&request_id);
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
                // The scheduler has now dropped the failed/cancelled operation.
                // Preserve allocation failure as the terminal cause rather than
                // exposing the cancellation used to clean up the operation.
                // Cancellation IDs were normalized at the provider boundary;
                // scheduler cancellations supplied the explicit ID above.
                pending.bytes = Vec::new();
                let error = pending.allocation_error.take().unwrap_or(error);
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
                // Lazily reserve the full declaration on the first
                // chunk, avoiding geometric aggregate growth/copies. Even the
                // legacy constructors use fallible aggregation. The scheduler
                // has already validated this chunk against the declared length.
                let reservation = if pending.bytes.capacity() == 0 {
                    usize::try_from(pending.request.manifest().io_bytes)
                        .map_err(|_| internal_error("declared bytes do not fit usize"))
                        .and_then(|declared| {
                            pending.bytes.try_reserve_exact(declared).map_err(|error| {
                                IoError::Provider {
                                    message: format!("cell aggregation allocation failed: {error}"),
                                }
                            })
                        })
                } else {
                    pending
                        .bytes
                        .try_reserve_exact(chunk.bytes().len())
                        .map_err(|error| IoError::Provider {
                            message: format!("cell aggregation allocation failed: {error}"),
                        })
                };
                if let Err(error) = reservation {
                    pending.bytes = Vec::new();
                    pending.allocation_error = Some(error);
                    // Do not emit a terminal event (and release admission) until
                    // scheduler.poll has removed this cancelled operation.
                    if !self.scheduler.cancel(request_id) {
                        return Poll::Ready(Err(internal_error(
                            "failed aggregate has no active operation",
                        )));
                    }
                    context.waker().wake_by_ref();
                    return Poll::Pending;
                }
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

// Only Arc handles move under the notification lock. Foreign Waker callbacks,
// including clone and final drop, always run after both metadata locks release.
struct RegisteredWaker(Option<Waker>);

impl RegisteredWaker {
    fn wake(&self) {
        if let Some(waker) = &self.0 {
            let _ = catch_unwind(AssertUnwindSafe(|| waker.wake_by_ref()));
        }
    }
}

impl Drop for RegisteredWaker {
    fn drop(&mut self) {
        if let Some(waker) = self.0.take() {
            let _ = catch_unwind(AssertUnwindSafe(|| drop(waker)));
        }
    }
}

#[derive(Default)]
struct Waiters {
    // Public polling has one consumer; changing its Context replaces this slot.
    public: Option<Arc<RegisteredWaker>>,
    // A blocking reader retains its own slot, even when contexts share a Waker.
    blocking: BTreeMap<IoRequestId, Arc<RegisteredWaker>>,
}

impl Waiters {
    fn wake(self) {
        if let Some(waiter) = self.public {
            waiter.wake();
        }
        for waiter in self.blocking.into_values() {
            waiter.wake();
        }
    }
}

#[derive(Default)]
struct WakeState {
    waiters: Waiters,
    busy: usize,
    notified: bool,
    dispatching: bool,
}

impl WakeState {
    fn take_notification(&mut self) -> Option<Waiters> {
        if self.busy != 0 || self.dispatching || !self.notified {
            return None;
        }
        self.notified = false;
        self.dispatching = true;
        Some(core::mem::take(&mut self.waiters))
    }
}

#[derive(Default)]
struct SchedulerWake {
    state: Mutex<WakeState>,
}

struct Registration {
    target: Option<IoRequestId>,
    waiter: Arc<RegisteredWaker>,
}

struct WakeBusy<'a> {
    wake: &'a SchedulerWake,
}

impl Drop for WakeBusy<'_> {
    fn drop(&mut self) {
        let waiters = {
            let mut state = self.wake.lock();
            state.busy -= 1;
            state.take_notification()
        };
        self.wake.dispatch(waiters);
    }
}

impl SchedulerWake {
    fn lock(&self) -> MutexGuard<'_, WakeState> {
        self.state.lock().unwrap_or_else(|error| error.into_inner())
    }

    fn register(&self, target: Option<IoRequestId>, waker: &Waker) -> Registration {
        let waiter = Arc::new(RegisteredWaker(Some(waker.clone())));
        let retained = Arc::clone(&waiter);
        let previous = {
            let mut state = self.lock();
            match target {
                None => state.waiters.public.replace(retained),
                Some(id) => state.waiters.blocking.insert(id, retained),
            }
        };
        drop(previous);
        Registration { target, waiter }
    }

    fn unregister(&self, registration: &Registration) {
        let removed = {
            let mut state = self.lock();
            let current = match registration.target {
                None => state.waiters.public.as_ref(),
                Some(id) => state.waiters.blocking.get(&id),
            };
            // A wake/drop callback may have registered a newer poll already.
            if current.is_some_and(|waiter| Arc::ptr_eq(waiter, &registration.waiter)) {
                match registration.target {
                    None => state.waiters.public.take(),
                    Some(id) => state.waiters.blocking.remove(&id),
                }
            } else {
                None
            }
        };
        drop(removed);
    }

    fn begin_busy(&self) -> WakeBusy<'_> {
        self.lock().busy += 1;
        WakeBusy { wake: self }
    }

    fn notify(&self) {
        let waiters = {
            let mut state = self.lock();
            state.notified = true;
            state.take_notification()
        };
        self.dispatch(waiters);
    }

    fn dispatch(&self, waiters: Option<Waiters>) {
        let Some(mut waiters) = waiters else {
            return;
        };
        loop {
            waiters.wake();
            let next = {
                let mut state = self.lock();
                state.dispatching = false;
                // A concurrent worker or reentrant callback may have notified
                // during dispatch. Taking the next batch under this same lock
                // prevents a lost wake at the busy/dispatch handoff.
                state.take_notification()
            };
            match next {
                Some(next) => waiters = next,
                None => return,
            }
        }
    }
}

impl Wake for SchedulerWake {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.notify();
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
#[path = "io_budget_tests.rs"]
mod budget_tests;

#[cfg(test)]
#[path = "io_waker_tests.rs"]
mod waker_tests;

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
