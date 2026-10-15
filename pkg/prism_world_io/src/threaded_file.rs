//! Bounded, acknowledgement-driven filesystem work. Only workers own file handles.

use std::{
    collections::VecDeque,
    fs::File,
    io::{self, Read, Seek, SeekFrom},
    panic::{catch_unwind, AssertUnwindSafe},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Condvar, Mutex, MutexGuard,
    },
    task::{Context, Poll, Waker},
    thread,
};

use crate::{ChunkedRead, IoError, ReadChunk, ReadProvider, ReadRequest, RequestId};

/// Hard limits shared by all clones of a [`ThreadedFileProvider`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ThreadedFileConfig {
    /// Fixed number of workers; no replacement or per-request threads are spawned.
    pub workers: usize,
    /// Maximum admitted operations, including parked results and cancelled reads
    /// that have not yet returned from the OS. Also bounds the ready queue.
    pub capacity: usize,
    /// Maximum payload per operation. Requests' chunk sizes are capped at this limit.
    pub max_chunk_size: usize,
}

impl Default for ThreadedFileConfig {
    fn default() -> Self {
        Self {
            workers: 2,
            capacity: 64,
            max_chunk_size: 64 * 1024,
        }
    }
}

/// A cloneable filesystem provider with a fixed worker pool and bounded admission.
///
/// `start`, poll, cancellation and destruction never open, seek, read or close a
/// file, wait for queue space, or join workers. They do use short metadata locks;
/// this is not a lock-free or hard real-time API. Construction starts all workers.
/// Dropping the last provider clone initiates shutdown, even if operations survive.
#[derive(Clone)]
pub struct ThreadedFileProvider {
    owner: Arc<Owner>,
}

struct Owner {
    shared: Arc<Shared>,
}

struct Shared {
    path: PathBuf,
    config: ThreadedFileConfig,
    state: Mutex<PoolState>,
    ready: Condvar,
    workers_alive: AtomicUsize,
    #[cfg(test)]
    before_io: Mutex<Option<Arc<dyn Fn() + Send + Sync>>>,
}

struct PoolState {
    closed: bool,
    // These collections reserve their entire configured capacity at construction.
    slots: Vec<Option<Entry>>,
    queue: VecDeque<usize>,
}

struct Control {
    cancelled: AtomicBool,
}

struct Entry {
    control: Arc<Control>,
    work: Option<Work>,
    phase: Phase,
    waker: Option<Waker>,
}

enum Phase {
    Queued,
    Running,
    Ready(Result<Option<ReadChunk>, IoError>),
    // A cancelled parked result stays charged to its slot until a worker has
    // destroyed its payload and file. It must not be dropped by the caller.
    CleanupReady(Result<Option<ReadChunk>, IoError>),
    AwaitingAck,
}

struct Work {
    request: ReadRequest,
    request_id: RequestId,
    file: Option<File>,
    delivered: u64,
    chunk_size: usize,
}

/// One bounded stream. A subsequent poll acknowledges the previous chunk and
/// queues the next worker turn. Dropping it requests cancellation without joining.
pub struct ThreadedFileReadOperation {
    shared: Arc<Shared>,
    control: Arc<Control>,
    slot: usize,
    request_id: RequestId,
    finished: bool,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    // No user code or filesystem work executes while a metadata lock is held.
    mutex.lock().unwrap_or_else(|error| error.into_inner())
}

fn provider_error(error: impl std::fmt::Display) -> IoError {
    IoError::Provider {
        message: error.to_string(),
    }
}

fn allocate_chunk(length: usize) -> Result<Vec<u8>, IoError> {
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(length).map_err(provider_error)?;
    bytes.resize(length, 0);
    Ok(bytes)
}

fn wake(waker: Option<Waker>) {
    if let Some(waker) = waker {
        // A host callback must not kill a fixed worker or prevent shutdown of
        // the remaining operations. The callback runs without metadata locks.
        let _ = catch_unwind(AssertUnwindSafe(|| waker.wake()));
    }
}

fn discard_waker(waker: Option<Waker>) {
    let _ = catch_unwind(AssertUnwindSafe(|| drop(waker)));
}

fn queue_cleanup(entry: &mut Entry) -> bool {
    match entry.phase {
        Phase::Ready(_) => {
            let Phase::Ready(result) = std::mem::replace(&mut entry.phase, Phase::Running) else {
                unreachable!();
            };
            entry.phase = Phase::CleanupReady(result);
            true
        }
        Phase::AwaitingAck => {
            entry.phase = Phase::Queued;
            true
        }
        _ => false,
    }
}

impl ThreadedFileProvider {
    /// Starts a pool without touching `path`. Invalid limits, reservation failures
    /// and the original OS thread-spawn error are returned to the caller.
    pub fn new(path: impl Into<PathBuf>, config: ThreadedFileConfig) -> io::Result<Self> {
        if config.workers == 0
            || config.capacity == 0
            || config.max_chunk_size == 0
            || config.max_chunk_size > isize::MAX as usize
            || config.capacity.checked_mul(config.max_chunk_size).is_none()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "worker count, capacity and chunk limit must be positive and payload bound must fit usize/isize",
            ));
        }
        let mut slots = Vec::new();
        slots
            .try_reserve_exact(config.capacity)
            .map_err(io::Error::other)?;
        slots.resize_with(config.capacity, || None);
        let mut queue = VecDeque::new();
        queue
            .try_reserve_exact(config.capacity)
            .map_err(io::Error::other)?;
        let shared = Arc::new(Shared {
            path: path.into(),
            config,
            state: Mutex::new(PoolState {
                closed: false,
                slots,
                queue,
            }),
            ready: Condvar::new(),
            workers_alive: AtomicUsize::new(0),
            #[cfg(test)]
            before_io: Mutex::new(None),
        });
        for index in 0..config.workers {
            let worker_shared = Arc::clone(&shared);
            shared.workers_alive.fetch_add(1, Ordering::AcqRel);
            match thread::Builder::new()
                .name(format!("prism-file-{index}"))
                .spawn(move || worker_loop(worker_shared))
            {
                // Detach: shutdown must never wait for a potentially stuck syscall.
                Ok(handle) => drop(handle),
                Err(error) => {
                    shared.workers_alive.fetch_sub(1, Ordering::AcqRel);
                    shared.shutdown();
                    return Err(error);
                }
            }
        }
        Ok(Self {
            owner: Arc::new(Owner { shared }),
        })
    }

    /// Returns the source path, without performing filesystem operations.
    pub fn path(&self) -> &Path {
        &self.owner.shared.path
    }

    /// Returns the pool's immutable limits.
    pub fn config(&self) -> ThreadedFileConfig {
        self.owner.shared.config
    }

    /// Cancels admitted operations and rejects further starts across all clones.
    /// Idempotent and non-joining; already blocked OS calls cannot be interrupted.
    pub fn shutdown(&self) {
        self.owner.shared.shutdown();
    }

    /// Reports whether shutdown has begun (not whether workers have exited).
    pub fn is_shutdown(&self) -> bool {
        lock(&self.owner.shared.state).closed
    }

    /// Nonblocking observation that every worker has exited after shutdown.
    pub fn is_shutdown_complete(&self) -> bool {
        self.is_shutdown() && self.owner.shared.workers_alive.load(Ordering::Acquire) == 0
    }

    /// Counts occupied slots, including undelivered results and cancellation cleanup.
    pub fn outstanding(&self) -> usize {
        lock(&self.owner.shared.state)
            .slots
            .iter()
            .flatten()
            .count()
    }
}

impl ReadProvider for ThreadedFileProvider {
    type Operation = ThreadedFileReadOperation;

    fn start(
        &self,
        request: &ReadRequest,
        request_id: RequestId,
    ) -> Result<Self::Operation, IoError> {
        request.validate()?;
        let shared = &self.owner.shared;
        let mut state = lock(&shared.state);
        if state.closed {
            return Err(provider_error("threaded file provider is shut down"));
        }
        let slot = state
            .slots
            .iter()
            .position(Option::is_none)
            .ok_or(IoError::Backpressure {
                capacity: shared.config.capacity,
            })?;
        let control = Arc::new(Control {
            cancelled: AtomicBool::new(false),
        });
        state.slots[slot] = Some(Entry {
            control: Arc::clone(&control),
            work: Some(Work {
                request: request.clone(),
                request_id,
                file: None,
                delivered: 0,
                chunk_size: request.chunk_size().min(shared.config.max_chunk_size),
            }),
            phase: Phase::Queued,
            waker: None,
        });
        state.queue.push_back(slot);
        drop(state);
        shared.ready.notify_one();
        Ok(ThreadedFileReadOperation {
            shared: Arc::clone(shared),
            control,
            slot,
            request_id,
            finished: false,
        })
    }
}

impl ChunkedRead for ThreadedFileReadOperation {
    fn poll_chunk(
        &mut self,
        context: &mut Context<'_>,
    ) -> Poll<Result<Option<ReadChunk>, IoError>> {
        if self.control.cancelled.load(Ordering::Acquire) {
            return Poll::Ready(Err(IoError::Cancelled {
                request_id: self.request_id,
            }));
        }
        if self.finished {
            return Poll::Ready(Ok(None));
        }
        // RawWaker clone/drop may execute arbitrary host code. In particular,
        // clone can reenter this provider, so it must happen before locking.
        let new_waker = context.waker().clone();
        // Register and inspect readiness under the very same lock used to publish.
        // A completion before registration is observed; one after it takes this waker.
        let mut state = lock(&self.shared.state);
        if self.control.cancelled.load(Ordering::Acquire) {
            return Poll::Ready(Err(IoError::Cancelled {
                request_id: self.request_id,
            }));
        }
        let entry = state.slots[self.slot]
            .as_mut()
            .expect("live operation slot");
        debug_assert!(Arc::ptr_eq(&entry.control, &self.control));
        if matches!(entry.phase, Phase::Ready(_)) {
            let Phase::Ready(result) = std::mem::replace(&mut entry.phase, Phase::AwaitingAck)
            else {
                unreachable!();
            };
            let old_waker = entry.waker.take();
            if entry.work.is_none() {
                // The worker has already closed the file on terminal publication.
                state.slots[self.slot] = None;
                self.finished = true;
            }
            drop(state);
            discard_waker(old_waker);
            discard_waker(Some(new_waker));
            return Poll::Ready(result);
        }
        let old_waker = entry.waker.replace(new_waker);
        let queue = matches!(entry.phase, Phase::AwaitingAck);
        if queue {
            entry.phase = Phase::Queued;
            state.queue.push_back(self.slot);
        }
        drop(state);
        discard_waker(old_waker);
        if queue {
            self.shared.ready.notify_one();
        }
        Poll::Pending
    }

    fn cancel(&mut self) {
        if !self.finished {
            self.shared.cancel(self.slot, &self.control);
        }
    }
}

impl Drop for ThreadedFileReadOperation {
    fn drop(&mut self) {
        self.cancel();
    }
}

impl Drop for Owner {
    fn drop(&mut self) {
        self.shared.shutdown();
    }
}

impl Shared {
    fn cancel(&self, slot: usize, control: &Arc<Control>) {
        let mut state = lock(&self.state);
        let Some(entry) = state.slots[slot]
            .as_mut()
            .filter(|e| Arc::ptr_eq(&e.control, control))
        else {
            return;
        };
        control.cancelled.store(true, Ordering::Release);
        let waker = entry.waker.take();
        if queue_cleanup(entry) {
            // Cleanup also runs on a worker, keeping file destruction off callers.
            state.queue.push_back(slot);
        }
        drop(state);
        self.ready.notify_all();
        wake(waker);
    }

    fn shutdown(&self) {
        // Take each waker outside the lock without allocating an unbounded list.
        let mut state = lock(&self.state);
        state.closed = true;
        for slot in 0..state.slots.len() {
            let Some(entry) = state.slots[slot].as_mut() else {
                continue;
            };
            entry.control.cancelled.store(true, Ordering::Release);
            if queue_cleanup(entry) {
                state.queue.push_back(slot);
            }
        }
        drop(state);
        self.ready.notify_all();
        for slot in 0..self.config.capacity {
            let waker = lock(&self.state).slots[slot]
                .as_mut()
                .and_then(|e| e.waker.take());
            wake(waker);
        }
    }
}

impl Work {
    fn read_chunk(&mut self, path: &Path, control: &Control) -> Result<Option<ReadChunk>, IoError> {
        self.check_cancel(control)?;
        if self.delivered == self.request.length() {
            return Ok(None);
        }
        if self.file.is_none() {
            let mut file = File::open(path).map_err(provider_error)?;
            self.check_cancel(control)?;
            file.seek(SeekFrom::Start(self.request.offset()))
                .map_err(provider_error)?;
            self.file = Some(file);
        }
        self.check_cancel(control)?;
        let length = (self.request.length() - self.delivered).min(self.chunk_size as u64) as usize;
        let mut bytes = allocate_chunk(length)?;
        let file = self.file.as_mut().expect("worker opened file");
        let mut actual = 0;
        while actual < length {
            if control.cancelled.load(Ordering::Acquire) {
                return Err(IoError::Cancelled {
                    request_id: self.request_id,
                });
            }
            match file.read(&mut bytes[actual..]) {
                Ok(0) => {
                    return Err(IoError::ShortRead {
                        expected: self.request.length(),
                        actual: self.delivered + actual as u64,
                    })
                }
                Ok(read) => actual += read,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(provider_error(error)),
            }
        }
        let offset = self.request.offset() + self.delivered;
        self.delivered += length as u64;
        Ok(Some(ReadChunk::new(offset, bytes)))
    }

    fn check_cancel(&self, control: &Control) -> Result<(), IoError> {
        if control.cancelled.load(Ordering::Acquire) {
            Err(IoError::Cancelled {
                request_id: self.request_id,
            })
        } else {
            Ok(())
        }
    }
}

struct WorkerExit(Arc<Shared>);

impl Drop for WorkerExit {
    fn drop(&mut self) {
        self.0.workers_alive.fetch_sub(1, Ordering::AcqRel);
    }
}

fn worker_loop(shared: Arc<Shared>) {
    let _exit = WorkerExit(Arc::clone(&shared));
    loop {
        let (slot, control, mut work, previous_phase) = {
            let mut state = lock(&shared.state);
            loop {
                if let Some(slot) = state.queue.pop_front() {
                    let entry = state.slots[slot].as_mut().expect("queued slot");
                    let control = Arc::clone(&entry.control);
                    let work = entry.work.take();
                    let previous_phase = std::mem::replace(&mut entry.phase, Phase::Running);
                    break (slot, control, work, previous_phase);
                }
                if state.closed {
                    return;
                }
                state = shared
                    .ready
                    .wait(state)
                    .unwrap_or_else(|error| error.into_inner());
            }
        };
        // A cancelled ready buffer is freed on the worker while the slot is
        // still occupied, before another request can allocate under this quota.
        if let Phase::CleanupReady(result) = previous_phase {
            drop(result);
        }
        let mut result = None;
        if !control.cancelled.load(Ordering::Acquire) {
            result = Some(
                catch_unwind(AssertUnwindSafe(|| {
                    #[cfg(test)]
                    {
                        let hook = lock(&shared.before_io).clone();
                        if let Some(hook) = hook {
                            hook();
                        }
                    }
                    work.as_mut()
                        .expect("queued read work")
                        .read_chunk(&shared.path, &control)
                }))
                .unwrap_or_else(|_| Err(provider_error("file worker read panicked"))),
            );
        }
        let terminal = match (&result, &work) {
            (Some(Ok(Some(_))), Some(work)) => work.delivered == work.request.length(),
            _ => true,
        };
        if terminal {
            // Even file close/destruction happens outside all metadata locks.
            drop(work.take());
        }
        let mut state = lock(&shared.state);
        let entry = state.slots[slot].as_mut().expect("worker owns slot");
        let waker = entry.waker.take();
        if control.cancelled.load(Ordering::Acquire) {
            // Keep the Running slot reserved while destroying a raced result
            // and nonterminal file. No new read can overlap their payload.
            drop(state);
            drop(result);
            drop(work);
            let mut state = lock(&shared.state);
            state.slots[slot] = None;
            drop(state);
        } else {
            entry.work = work.take();
            entry.phase = Phase::Ready(result.expect("noncancelled worker result"));
            drop(state);
        }
        wake(waker);
    }
}

#[cfg(test)]
pub(super) mod test_support {
    use super::*;

    pub fn set_hook(provider: &ThreadedFileProvider, hook: Arc<dyn Fn() + Send + Sync>) {
        *lock(&provider.owner.shared.before_io) = Some(hook);
    }

    pub fn allocated_slot_count(provider: &ThreadedFileProvider) -> usize {
        lock(&provider.owner.shared.state).slots.len()
    }

    pub fn allocation_error(length: usize) -> Result<Vec<u8>, IoError> {
        allocate_chunk(length)
    }

    pub fn wait_ready(provider: &ThreadedFileProvider) {
        let end = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            let ready = lock(&provider.owner.shared.state)
                .slots
                .iter()
                .flatten()
                .any(|entry| matches!(entry.phase, Phase::Ready(_)));
            if ready {
                return;
            }
            assert!(
                std::time::Instant::now() < end,
                "worker did not publish a result"
            );
            thread::yield_now();
        }
    }
}
