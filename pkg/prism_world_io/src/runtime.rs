use std::{
    cmp::Ordering as CompareOrdering,
    collections::BinaryHeap,
    fmt,
    future::Future,
    io,
    pin::Pin,
    sync::{Arc, Condvar, Mutex, MutexGuard},
    task::{Context, Poll, Waker},
    thread::{self, JoinHandle},
};

use crate::{
    validate_chunk, AsyncChunkReader, CancellationToken, Chunk, ChunkProvider, ReadFuture,
    ReadRequest,
};

/// Configuration for the standard-library I/O runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IoConfig {
    /// Number of worker threads.
    pub workers: usize,
    /// Maximum number of queued or running requests.
    pub max_in_flight: usize,
}

impl Default for IoConfig {
    fn default() -> Self {
        Self {
            workers: 1,
            max_in_flight: 64,
        }
    }
}

/// Runtime construction failures.
#[derive(Debug)]
pub enum RuntimeError {
    /// A runtime needs at least one worker.
    InvalidWorkerCount,
    /// A runtime needs positive capacity.
    InvalidCapacity,
    /// A worker thread could not be started.
    WorkerSpawn {
        /// Platform-independent reason reported by thread creation.
        kind: io::ErrorKind,
    },
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidWorkerCount => {
                formatter.write_str("I/O runtime needs at least one worker")
            }
            Self::InvalidCapacity => formatter.write_str("I/O runtime needs positive capacity"),
            Self::WorkerSpawn { kind } => write!(formatter, "I/O worker could not start: {kind}"),
        }
    }
}

impl std::error::Error for RuntimeError {}

/// Submission failures caused by runtime state or bounded capacity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubmitError {
    /// The bounded queue already contains its maximum in-flight requests.
    Backpressure {
        /// Maximum number of queued or running requests.
        capacity: usize,
    },
    /// The runtime has begun shutting down.
    Closed,
}

impl fmt::Display for SubmitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Backpressure { capacity } => {
                write!(formatter, "I/O queue back-pressure at capacity {capacity}")
            }
            Self::Closed => formatter.write_str("I/O runtime is closed"),
        }
    }
}

impl std::error::Error for SubmitError {}

/// Errors returned by a submitted chunk read.
#[derive(Debug)]
pub enum IoError<E> {
    /// The request was cancelled before completion.
    Cancelled,
    /// The provider returned a byte count different from the request.
    ShortRead {
        /// Number of bytes requested.
        expected: usize,
        /// Number of bytes returned.
        actual: usize,
    },
    /// The provider returned bytes with a different checksum.
    ChecksumMismatch {
        /// Checksum supplied by the request.
        expected: u32,
        /// Checksum computed from the returned bytes.
        actual: u32,
    },
    /// The provider returned its own error.
    Provider(E),
    /// Submission failed before a worker could start the request.
    Submit(SubmitError),
}

impl<E: fmt::Display> fmt::Display for IoError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled => formatter.write_str("I/O request cancelled"),
            Self::ShortRead { expected, actual } => {
                write!(
                    formatter,
                    "short read: expected {expected} bytes, got {actual}"
                )
            }
            Self::ChecksumMismatch { expected, actual } => {
                write!(
                    formatter,
                    "checksum mismatch: expected {expected:#010x}, got {actual:#010x}"
                )
            }
            Self::Provider(error) => write!(formatter, "provider error: {error}"),
            Self::Submit(error) => write!(formatter, "submission error: {error}"),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for IoError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Provider(error) => Some(error),
            Self::Submit(error) => Some(error),
            Self::Cancelled | Self::ShortRead { .. } | Self::ChecksumMismatch { .. } => None,
        }
    }
}

struct ResultCell<E> {
    state: Mutex<ResultState<E>>,
}

struct ResultState<E> {
    result: Option<Result<Chunk, IoError<E>>>,
    waker: Option<Waker>,
}

impl<E> ResultCell<E> {
    fn new() -> Self {
        Self {
            state: Mutex::new(ResultState {
                result: None,
                waker: None,
            }),
        }
    }
}

struct QueuedJob<E> {
    request: ReadRequest,
    cancel: CancellationToken,
    result: Arc<ResultCell<E>>,
    sequence: u64,
}

impl<E> PartialEq for QueuedJob<E> {
    fn eq(&self, other: &Self) -> bool {
        self.sequence == other.sequence
    }
}

impl<E> Eq for QueuedJob<E> {}

impl<E> PartialOrd for QueuedJob<E> {
    fn partial_cmp(&self, other: &Self) -> Option<CompareOrdering> {
        Some(self.cmp(other))
    }
}

impl<E> Ord for QueuedJob<E> {
    fn cmp(&self, other: &Self) -> CompareOrdering {
        self.request
            .priority()
            .cmp(&other.request.priority())
            .then_with(|| other.sequence.cmp(&self.sequence))
    }
}

struct RuntimeState<E> {
    queue: BinaryHeap<QueuedJob<E>>,
    in_flight: usize,
    next_sequence: u64,
    shutdown: bool,
}

struct Shared<P>
where
    P: ChunkProvider,
{
    provider: Arc<P>,
    state: Mutex<RuntimeState<P::Error>>,
    wake: Condvar,
    capacity: usize,
}

/// A bounded worker-thread runtime for prioritized chunk reads.
pub struct IoRuntime<P>
where
    P: ChunkProvider,
{
    shared: Arc<Shared<P>>,
    workers: Vec<JoinHandle<()>>,
}

impl<P> IoRuntime<P>
where
    P: ChunkProvider + Send + Sync + 'static,
    P::Error: Send + 'static,
{
    /// Starts the configured worker threads.
    pub fn new(provider: P, config: IoConfig) -> Result<Self, RuntimeError> {
        if config.workers == 0 {
            return Err(RuntimeError::InvalidWorkerCount);
        }
        if config.max_in_flight == 0 {
            return Err(RuntimeError::InvalidCapacity);
        }

        let shared = Arc::new(Shared {
            provider: Arc::new(provider),
            state: Mutex::new(RuntimeState {
                queue: BinaryHeap::new(),
                in_flight: 0,
                next_sequence: 0,
                shutdown: false,
            }),
            wake: Condvar::new(),
            capacity: config.max_in_flight,
        });
        let mut workers = Vec::with_capacity(config.workers);
        for index in 0..config.workers {
            let worker_shared = Arc::clone(&shared);
            let result = thread::Builder::new()
                .name(format!("prism-world-io-{index}"))
                .spawn(move || worker_loop(worker_shared));
            match result {
                Ok(worker) => workers.push(worker),
                Err(error) => {
                    shutdown_shared(&shared);
                    for worker in workers {
                        let _ = worker.join();
                    }
                    return Err(RuntimeError::WorkerSpawn { kind: error.kind() });
                }
            }
        }
        Ok(Self { shared, workers })
    }

    /// Submits a request, returning a pollable ticket or an immediate
    /// back-pressure/closed error.
    pub fn submit(&self, request: ReadRequest) -> Result<ReadTicket<P::Error>, SubmitError> {
        let cancel = CancellationToken::new();
        let result = Arc::new(ResultCell::new());
        let mut state = lock(&self.shared.state);
        if state.shutdown {
            return Err(SubmitError::Closed);
        }
        if state.in_flight >= self.shared.capacity {
            return Err(SubmitError::Backpressure {
                capacity: self.shared.capacity,
            });
        }
        let sequence = state.next_sequence;
        state.next_sequence = state.next_sequence.wrapping_add(1);
        state.in_flight += 1;
        state.queue.push(QueuedJob {
            request,
            cancel: cancel.clone(),
            result: Arc::clone(&result),
            sequence,
        });
        drop(state);
        self.shared.wake.notify_one();
        Ok(ReadTicket { cancel, result })
    }

    /// Returns the number of queued or running requests.
    #[must_use]
    pub fn in_flight(&self) -> usize {
        lock(&self.shared.state).in_flight
    }
}

impl<P> AsyncChunkReader for IoRuntime<P>
where
    P: ChunkProvider + Send + Sync + 'static,
    P::Error: Send + 'static,
{
    type Error = IoError<P::Error>;

    fn read_chunk<'a>(&'a self, request: ReadRequest) -> ReadFuture<'a, Self::Error> {
        Box::pin(async move {
            let ticket = self.submit(request).map_err(IoError::Submit)?;
            ticket.await
        })
    }
}

impl<P> Drop for IoRuntime<P>
where
    P: ChunkProvider,
{
    fn drop(&mut self) {
        let drained = {
            let mut state = lock(&self.shared.state);
            state.shutdown = true;
            let jobs: Vec<_> = state.queue.drain().collect();
            state.in_flight = state.in_flight.saturating_sub(jobs.len());
            jobs
        };
        for job in drained {
            complete(&job.result, Err(IoError::Cancelled));
        }
        self.shared.wake.notify_all();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

/// A pollable, cancellable result ticket returned by [`IoRuntime::submit`].
pub struct ReadTicket<E> {
    cancel: CancellationToken,
    result: Arc<ResultCell<E>>,
}

impl<E> ReadTicket<E> {
    /// Cancels this request and wakes its current poller.
    pub fn cancel(&self) {
        self.cancel.cancel();
        let waker = {
            let mut state = lock(&self.result.state);
            state.waker.take()
        };
        if let Some(waker) = waker {
            waker.wake();
        }
    }

    /// Returns whether cancellation has been requested.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancel.is_cancelled()
    }
}

impl<E> Future for ReadTicket<E> {
    type Output = Result<Chunk, IoError<E>>;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        if self.cancel.is_cancelled() {
            return Poll::Ready(Err(IoError::Cancelled));
        }
        let mut state = lock(&self.result.state);
        if let Some(result) = state.result.take() {
            return Poll::Ready(result);
        }
        state.waker = Some(context.waker().clone());
        if self.cancel.is_cancelled() {
            state.waker.take();
            Poll::Ready(Err(IoError::Cancelled))
        } else {
            Poll::Pending
        }
    }
}

fn worker_loop<P>(shared: Arc<Shared<P>>)
where
    P: ChunkProvider + Send + Sync + 'static,
    P::Error: Send + 'static,
{
    loop {
        let job = {
            let mut state = lock(&shared.state);
            while state.queue.is_empty() && !state.shutdown {
                state = wait(&shared.wake, state);
            }
            if state.shutdown {
                return;
            }
            state.queue.pop()
        };
        let Some(job) = job else {
            continue;
        };
        let result = if job.cancel.is_cancelled() {
            Err(IoError::Cancelled)
        } else {
            match shared.provider.read_chunk(job.request.range(), &job.cancel) {
                Err(error) => Err(IoError::Provider(error)),
                Ok(bytes) if job.cancel.is_cancelled() => Err(IoError::Cancelled),
                Ok(bytes) => match validate_chunk(
                    job.request.range(),
                    &bytes,
                    job.request.expected_checksum(),
                ) {
                    Ok(()) => Ok(Chunk {
                        range: job.request.range(),
                        bytes,
                    }),
                    Err(crate::ChunkValidationError::ShortRead { expected, actual }) => {
                        Err(IoError::ShortRead { expected, actual })
                    }
                    Err(crate::ChunkValidationError::ChecksumMismatch { expected, actual }) => {
                        Err(IoError::ChecksumMismatch { expected, actual })
                    }
                },
            }
        };
        complete(&job.result, result);
        let mut state = lock(&shared.state);
        state.in_flight = state.in_flight.saturating_sub(1);
        shared.wake.notify_all();
    }
}

fn complete<E>(cell: &Arc<ResultCell<E>>, result: Result<Chunk, IoError<E>>) {
    let waker = {
        let mut state = lock(&cell.state);
        if state.result.is_some() {
            None
        } else {
            state.result = Some(result);
            state.waker.take()
        }
    };
    if let Some(waker) = waker {
        waker.wake();
    }
}

fn shutdown_shared<P>(shared: &Arc<Shared<P>>)
where
    P: ChunkProvider,
{
    let mut state = lock(&shared.state);
    state.shutdown = true;
    shared.wake.notify_all();
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn wait<'a, T>(condition: &Condvar, guard: MutexGuard<'a, T>) -> MutexGuard<'a, T> {
    condition
        .wait(guard)
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}
