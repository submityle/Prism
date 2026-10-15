//! Native file reads with worker-owned descriptors and acknowledgement-driven turns.
//!
//! `IoReactor` owns buffers, **not descriptors**. Dropping its future does not
//! cancel a submitted read. Here only `NamedThreads::Io` jobs can own files or
//! reactor futures. They observe completion before closing a file, including
//! after cancellation. The reactor receives only an internal, non-panicking
//! thread-unpark waker, never a host waker.
//!
//! This module does not create threads or services. The host supplies one shared
//! reactor and existing named workers, and owns their potentially joining teardown.
//! Operations borrow those owners so dropping an operation/provider cannot drop
//! the last service handle on a compute thread. Already dispatched jobs retain
//! their own reactor reference until their native work and file cleanup finish.

use std::{
    collections::VecDeque,
    fs::File,
    future::Future,
    io,
    os::fd::AsRawFd,
    panic::{catch_unwind, AssertUnwindSafe},
    path::{Path, PathBuf},
    pin::Pin,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex, MutexGuard,
    },
    task::{Context, Poll, Wake, Waker},
    thread::{self, Thread},
};

use prism_platform::aio::IoPriority;
use prism_tasks::{AioIoError, AioReadFuture, IoReactor, NamedThreads, ReadBuf, ThreadCategory};

use crate::{ChunkedRead, IoError, ReadChunk, ReadProvider, ReadRequest, RequestId};

#[path = "reactor_queued.rs"]
mod queued;
pub use queued::{QueuedReactorFileProvider, QueuedReactorFileReadOperation};

/// Limits shared by all clones of one native file provider.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReactorFileConfig {
    /// Admitted streams, including ready results, ACK waits and cancelled cleanup.
    pub capacity: usize,
    /// Maximum bytes per native submission/result. Request chunk sizes are capped.
    pub max_chunk_size: usize,
    /// Maximum declared length of one request (not an RSS limit).
    pub max_request_bytes: u64,
    /// Sum of declared lengths of all charged streams (not an RSS limit).
    pub max_total_bytes: u64,
}

impl Default for ReactorFileConfig {
    fn default() -> Self {
        Self {
            capacity: 64,
            max_chunk_size: 64 * 1024,
            max_request_bytes: u64::MAX,
            max_total_bytes: u64::MAX,
        }
    }
}

impl ReactorFileConfig {
    fn validate(self) -> io::Result<()> {
        if self.capacity == 0
            || self.max_chunk_size == 0
            || self.max_chunk_size > isize::MAX as usize
            || self.capacity.checked_mul(self.max_chunk_size).is_none()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "native IO capacity/chunk size must be positive and their payload bound must fit usize/isize",
            ));
        }
        Ok(())
    }
}

/// A bounded `ReadProvider` using real native reads, not blocking `File::read`.
///
/// Construction/start/poll/cancel/drop do no filesystem calls, thread creation,
/// joins, capacity waits or native-completion waits. Short metadata locks are
/// used, so this is not a lock-free/hard-real-time API. The existing IO workers
/// open files, submit to the fixed shared reactor, park until completion, and
/// close files. At most one turn per admitted stream is queued/running, and no
/// next read is dispatched until the poll *after* a chunk acknowledges it.
///
/// All clones share request and byte quotas. Backpressure is an explicit error,
/// not an unbounded waiting list. A cancellation may be observed immediately,
/// but its quota stays charged until IO-worker cleanup has actually finished.
/// A completed result stays charged until consumed or dropped. Retained chunks
/// already handed to the caller are outside the provider's payload bound.
///
/// The host must reserve enough `NamedThreads::Io` capacity for progress: these
/// jobs park IO workers, never compute workers. Do not put jobs waiting for
/// these operations on the same saturated IO lane. Dropping a provider alone
/// does not cancel operations; each operation independently owns its admission.
/// The real platform backend is currently Apple-only; no fallback read is faked.
#[derive(Clone)]
pub struct ReactorFileProvider<'services> {
    shared: Arc<Shared>,
    reactor: &'services Arc<IoReactor>,
    workers: &'services NamedThreads,
}

/// One native stream. Drop requests cleanup without waiting for the OS.
pub struct ReactorFileReadOperation<'services> {
    provider: ReactorFileProvider<'services>,
    control: Arc<Control>,
    request_id: RequestId,
    slot: usize,
    finished: bool,
}

struct Shared {
    path: PathBuf,
    config: ReactorFileConfig,
    state: Mutex<State>,
    #[cfg(test)]
    hook: Mutex<Option<TestHook>>,
}

struct State {
    slots: Vec<Option<Entry>>,
    bytes: u64,
    admissions: Vec<Option<Admission>>,
    waiting: VecDeque<usize>,
}

struct Admission {
    control: Arc<Control>,
    work: Work,
    slot: Option<usize>,
    waker: Option<Waker>,
}

struct Control(AtomicBool);

struct Entry {
    control: Arc<Control>,
    charged_bytes: u64,
    work: Option<Work>,
    phase: Phase,
    waker: Option<Waker>,
}

enum Phase {
    Busy,
    Ready(Result<Option<ReadChunk>, IoError>),
    AwaitingAck,
}

// A File can be parked here between turns, but it is only ever destroyed by
// an IO job. Every path removing an admission first transfers Work to that job.
struct Work {
    request: ReadRequest,
    request_id: RequestId,
    chunk_size: usize,
    delivered: u64,
    file: Option<File>,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|error| error.into_inner())
}

fn error(message: impl std::fmt::Display) -> IoError {
    IoError::Provider {
        message: message.to_string(),
    }
}

// Foreign waker clone/drop/wake must never run under a metadata lock or unwind
// an IO worker. Clone happens before locking; replacement/destruction after it.
fn notify(waker: Option<Waker>) {
    if let Some(waker) = waker {
        let _ = catch_unwind(AssertUnwindSafe(|| waker.wake()));
    }
}

fn discard(waker: Option<Waker>) {
    let _ = catch_unwind(AssertUnwindSafe(|| drop(waker)));
}

fn notify_all(wakers: Vec<Waker>) {
    for waker in wakers {
        notify(Some(waker));
    }
}

impl<'services> ReactorFileProvider<'services> {
    /// Borrows host-owned services; never opens `path` or creates a service.
    /// Shut down the host services on a non-compute thread after dropping all
    /// providers/operations. Outstanding jobs still hold the reactor until drained.
    pub fn new(
        path: impl Into<PathBuf>,
        reactor: &'services Arc<IoReactor>,
        workers: &'services NamedThreads,
        config: ReactorFileConfig,
    ) -> io::Result<Self> {
        config.validate()?;
        if !IoReactor::supported() {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "native IO unavailable",
            ));
        }
        let mut slots = Vec::new();
        slots
            .try_reserve_exact(config.capacity)
            .map_err(io::Error::other)?;
        slots.resize_with(config.capacity, || None);
        let mut admissions = Vec::new();
        admissions
            .try_reserve_exact(config.capacity)
            .map_err(io::Error::other)?;
        admissions.resize_with(config.capacity, || None);
        let mut waiting = VecDeque::new();
        waiting
            .try_reserve_exact(config.capacity)
            .map_err(io::Error::other)?;
        Ok(Self {
            shared: Arc::new(Shared {
                path: path.into(),
                config,
                state: Mutex::new(State {
                    slots,
                    bytes: 0,
                    admissions,
                    waiting,
                }),
                #[cfg(test)]
                hook: Mutex::new(None),
            }),
            reactor,
            workers,
        })
    }

    /// Configured source, not opened until a worker's first nonempty turn.
    pub fn path(&self) -> &Path {
        &self.shared.path
    }

    /// Returns immutable limits shared by direct and queued clones.
    pub fn config(&self) -> ReactorFileConfig {
        self.shared.config
    }

    /// Adds bounded, wake-driven slot and byte-quota waiting.
    pub fn queued(self) -> QueuedReactorFileProvider<'services> {
        QueuedReactorFileProvider { provider: self }
    }

    fn validate_request(&self, request: &ReadRequest) -> Result<(), IoError> {
        request.validate()?;
        if request.length() != 0 && request.offset() + request.length() - 1 > i64::MAX as u64 {
            return Err(error("native read range exceeds signed file offsets"));
        }
        let config = self.shared.config;
        if request.length() > config.max_request_bytes || request.length() > config.max_total_bytes
        {
            return Err(error("native file request byte quota exceeded"));
        }
        Ok(())
    }

    /// Charged admissions, including cancellation cleanup and parked results.
    pub fn in_flight(&self) -> usize {
        lock(&self.shared.state).slots.iter().flatten().count()
    }

    /// Declared bytes charged until terminal consumption or completed cleanup.
    pub fn reserved_bytes(&self) -> u64 {
        lock(&self.shared.state).bytes
    }

    fn dispatch(&self, slot: usize, mut work: Option<Work>, previous: Phase) {
        let shared = Arc::clone(&self.shared);
        let reactor = Arc::clone(self.reactor);
        // Neither this closure nor Shared owns a NamedThreads clone, avoiding
        // last-owner lane joins on a lane worker (or on an operation's Drop).
        self.workers.dispatch(ThreadCategory::Io, move || {
            // In particular, cancelled ready payloads are destroyed on IO, while
            // still charged to the admission they were allocated under.
            drop(previous);
            let control = {
                let state = lock(&shared.state);
                Arc::clone(&state.slots[slot].as_ref().expect("dispatched slot").control)
            };
            let mut result = None;
            if !control.0.load(Ordering::Acquire) {
                result = Some(
                    catch_unwind(AssertUnwindSafe(|| {
                        work.as_mut()
                            .expect("read turn owns work")
                            .read(&shared, &reactor, &control)
                    }))
                    .unwrap_or_else(|_| Err(error("native file worker panicked"))),
                );
            }
            let terminal = match (&result, &work) {
                (Some(Ok(Some(_))), Some(work)) => work.delivered == work.request.length(),
                _ => true,
            };
            if terminal {
                close(&shared, &mut work);
            }
            let mut state = lock(&shared.state);
            let entry = state.slots[slot].as_mut().expect("worker owns slot");
            let waker = entry.waker.take();
            if control.0.load(Ordering::Acquire) {
                drop(state);
                drop(result);
                close(&shared, &mut work);
                // A slot cannot be reused until its file and payload are gone.
                let available = lock(&shared.state).release(slot, shared.config);
                notify_all(available);
            } else {
                entry.work = work.take();
                entry.phase = Phase::Ready(result.expect("uncancelled result"));
                drop(state);
            }
            notify(waker);
        });
    }
}

impl State {
    fn release(&mut self, slot: usize, config: ReactorFileConfig) -> Vec<Waker> {
        let entry = self.slots[slot].take().expect("charged slot");
        debug_assert!(entry.work.is_none());
        debug_assert!(entry.waker.is_none());
        self.bytes -= entry.charged_bytes;
        self.grant_waiters(config, None)
    }

    // Reserve both slot and bytes before taking the wake. No filesystem work,
    // payload allocation or host callback executes under this metadata lock.
    // A grant is inert until polled; canceling it can release it synchronously.
    fn grant_waiters(&mut self, config: ReactorFileConfig, polling: Option<usize>) -> Vec<Waker> {
        let mut wakers = Vec::new();
        while let Some(&index) = self.waiting.front() {
            let admission = self.admissions[index].as_ref().expect("waiting admission");
            let Some(bytes) = self
                .bytes
                .checked_add(admission.work.request.length())
                .filter(|bytes| *bytes <= config.max_total_bytes)
            else {
                break;
            };
            let Some(slot) = self.slots.iter().position(Option::is_none) else {
                break;
            };
            self.waiting.pop_front();
            let admission = self.admissions[index].as_mut().unwrap();
            self.slots[slot] = Some(Entry {
                control: Arc::clone(&admission.control),
                charged_bytes: admission.work.request.length(),
                work: None,
                phase: Phase::Busy,
                waker: None,
            });
            self.bytes = bytes;
            admission.slot = Some(slot);
            if polling != Some(index)
                && let Some(waker) = admission.waker.take()
            {
                wakers.push(waker);
            }
        }
        wakers
    }
}

impl<'services> ReadProvider for ReactorFileProvider<'services> {
    type Operation = ReactorFileReadOperation<'services>;

    fn start(
        &self,
        request: &ReadRequest,
        request_id: RequestId,
    ) -> Result<Self::Operation, IoError> {
        self.validate_request(request)?;
        let config = self.shared.config;
        let control = Arc::new(Control(AtomicBool::new(false)));
        let mut state = lock(&self.shared.state);
        let slot = state
            .slots
            .iter()
            .position(Option::is_none)
            .ok_or(IoError::Backpressure {
                capacity: config.capacity,
            })?;
        let bytes = state
            .bytes
            .checked_add(request.length())
            .filter(|bytes| *bytes <= config.max_total_bytes)
            .ok_or_else(|| error("native file total byte quota exceeded"))?;
        state.slots[slot] = Some(Entry {
            control: Arc::clone(&control),
            charged_bytes: request.length(),
            work: None,
            phase: Phase::Busy,
            waker: None,
        });
        state.bytes = bytes;
        drop(state);
        self.dispatch(
            slot,
            Some(Work {
                request: request.clone(),
                request_id,
                chunk_size: request.chunk_size().min(config.max_chunk_size),
                delivered: 0,
                file: None,
            }),
            Phase::Busy,
        );
        Ok(ReactorFileReadOperation {
            provider: self.clone(),
            control,
            request_id,
            slot,
            finished: false,
        })
    }
}

impl ChunkedRead for ReactorFileReadOperation<'_> {
    fn poll_chunk(
        &mut self,
        context: &mut Context<'_>,
    ) -> Poll<Result<Option<ReadChunk>, IoError>> {
        if self.control.0.load(Ordering::Acquire) {
            return Poll::Ready(Err(IoError::Cancelled {
                request_id: self.request_id,
            }));
        }
        if self.finished {
            return Poll::Ready(Ok(None));
        }
        let new_waker = context.waker().clone();
        let mut state = lock(&self.provider.shared.state);
        let entry = state.slots[self.slot]
            .as_mut()
            .expect("operation owns slot");
        let old_waker = entry.waker.take();
        if matches!(entry.phase, Phase::Ready(_)) {
            let Phase::Ready(result) = std::mem::replace(&mut entry.phase, Phase::AwaitingAck)
            else {
                unreachable!();
            };
            let available = if entry.work.is_none() {
                self.finished = true;
                state.release(self.slot, self.provider.shared.config)
            } else {
                Vec::new()
            };
            drop(state);
            notify_all(available);
            discard(old_waker);
            discard(Some(new_waker));
            return Poll::Ready(result);
        }
        entry.waker = Some(new_waker);
        let turn = if matches!(entry.phase, Phase::AwaitingAck) {
            entry.phase = Phase::Busy;
            Some(entry.work.take().expect("ACK owns parked work"))
        } else {
            None
        };
        drop(state);
        // Enqueue before disposing foreign wakers: even a hostile Drop cannot
        // strand an admission that has already transitioned to Busy.
        if let Some(work) = turn {
            self.provider.dispatch(self.slot, Some(work), Phase::Busy);
        }
        discard(old_waker);
        Poll::Pending
    }

    fn cancel(&mut self) {
        if self.finished || self.control.0.swap(true, Ordering::AcqRel) {
            return;
        }
        let mut state = lock(&self.provider.shared.state);
        // The running worker may already have observed cancellation and freed
        // this slot. Identity checks protect a new operation reusing its index.
        let Some(entry) = state.slots[self.slot]
            .as_mut()
            .filter(|entry| Arc::ptr_eq(&entry.control, &self.control))
        else {
            return;
        };
        let waker = entry.waker.take();
        let cleanup = if matches!(entry.phase, Phase::Busy) {
            None
        } else {
            Some((
                entry.work.take(),
                std::mem::replace(&mut entry.phase, Phase::Busy),
            ))
        };
        drop(state);
        if let Some((work, previous)) = cleanup {
            self.provider.dispatch(self.slot, work, previous);
        }
        notify(waker);
    }
}

impl Drop for ReactorFileReadOperation<'_> {
    fn drop(&mut self) {
        self.cancel();
    }
}

fn close(shared: &Shared, work: &mut Option<Work>) {
    if let Some(mut work) = work.take()
        && let Some(file) = work.file.take()
    {
        drop(file);
        #[cfg(test)]
        shared.event(TestEvent::Closed);
    }
    let _ = shared;
}

impl Work {
    fn read(
        &mut self,
        shared: &Shared,
        reactor: &IoReactor,
        control: &Control,
    ) -> Result<Option<ReadChunk>, IoError> {
        self.check_cancel(control)?;
        if self.delivered == self.request.length() {
            return Ok(None);
        }
        if self.file.is_none() {
            #[cfg(test)]
            shared.event(TestEvent::BeforeOpen);
            self.check_cancel(control)?;
            self.file = Some(File::open(&shared.path).map_err(error)?);
        }
        self.check_cancel(control)?;
        let length = (self.request.length() - self.delivered).min(self.chunk_size as u64) as usize;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(length).map_err(error)?;
        bytes.resize(length, 0);
        let offset = self.request.offset() + self.delivered;
        let file = self.file.as_ref().expect("IO worker opened file");
        // Build the infallible internal waker before submitting any native op.
        let waker = Waker::from(Arc::new(WorkerWake(thread::current())));
        let priority = native_priority(self.request.effective_lane());
        #[cfg(test)]
        shared.event(TestEvent::Priority(priority));
        let future =
            reactor.read_into(file.as_raw_fd(), offset, priority, bytes.into_boxed_slice());
        let mut pending = PendingRead {
            future,
            waker,
            done: false,
        };
        // Even a test-hook panic drains the native read before Work can unwind.
        #[cfg(test)]
        shared.event(TestEvent::Submitted);
        let read = pending.finish().map_err(error)?;
        #[cfg(test)]
        shared.event(TestEvent::Reaped {
            file_alive: file.metadata().is_ok(),
        });
        self.check_cancel(control)?;
        if read.bytes == 0 {
            return Err(IoError::ShortRead {
                expected: self.request.length(),
                actual: self.delivered,
            });
        }
        let mut bytes = read.data.into_vec();
        bytes.truncate(read.bytes);
        self.delivered += bytes.len() as u64;
        Ok(Some(ReadChunk::new(offset, bytes)))
    }

    fn check_cancel(&self, control: &Control) -> Result<(), IoError> {
        if control.0.load(Ordering::Acquire) {
            Err(IoError::Cancelled {
                request_id: self.request_id,
            })
        } else {
            Ok(())
        }
    }
}

struct WorkerWake(Thread);

impl Wake for WorkerWake {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}

// This guard only exists on IO workers. Its unwind path must not abandon a raw
// fd submission: finish observing native completion before Work drops its File.
struct PendingRead {
    future: AioReadFuture,
    waker: Waker,
    done: bool,
}

impl PendingRead {
    fn finish(&mut self) -> Result<ReadBuf, AioIoError> {
        loop {
            match Pin::new(&mut self.future).poll(&mut Context::from_waker(&self.waker)) {
                Poll::Ready(result) => {
                    self.done = true;
                    return result;
                }
                Poll::Pending => thread::park(),
            }
        }
    }
}

impl Drop for PendingRead {
    fn drop(&mut self) {
        if !self.done {
            let _ = self.finish();
        }
    }
}

fn native_priority(priority: prism_tasks::Priority) -> IoPriority {
    use prism_tasks::Priority;
    match priority {
        Priority::Background | Priority::Low => IoPriority::Low,
        Priority::Normal => IoPriority::Normal,
        Priority::High | Priority::Critical => IoPriority::High,
    }
}

#[cfg(test)]
type TestHook = Arc<dyn Fn(TestEvent) + Send + Sync>;

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum TestEvent {
    BeforeOpen,
    Priority(IoPriority),
    Submitted,
    Reaped { file_alive: bool },
    Closed,
}

#[cfg(test)]
impl Shared {
    fn event(&self, event: TestEvent) {
        let hook = lock(&self.hook).clone();
        if let Some(hook) = hook {
            hook(event);
        }
    }
}

#[cfg(test)]
pub(super) mod test_support {
    use super::*;

    pub fn set_hook(
        provider: &ReactorFileProvider<'_>,
        hook: Arc<dyn Fn(TestEvent) + Send + Sync>,
    ) {
        *lock(&provider.shared.hook) = Some(hook);
    }

    pub fn ready(provider: &ReactorFileProvider<'_>) -> bool {
        lock(&provider.shared.state)
            .slots
            .iter()
            .flatten()
            .any(|entry| matches!(entry.phase, Phase::Ready(_)))
    }

    pub fn validate(config: ReactorFileConfig) -> io::Result<()> {
        config.validate()
    }
}
