//! §24.5 Asynchronous I/O bridge: expose [`prism_platform`]'s batch async-read
//! queue ([`prism_platform::aio::AioQueue`]) as `Future`s that integrate with
//! this crate's work-stealing async executor.
//!
//! ## Why a reactor thread
//! [`AioQueue`] holds raw control-block pointers into the kernel, so it is
//! `!Send`: it must be created, used, and dropped on a single thread. The
//! bridge therefore owns one dedicated **reactor thread** that is the sole
//! owner of the queue for its entire lifetime. Callers talk to it only through
//! [`Submission`]s sent over an `mpsc` channel (every field of which is
//! `Send`), and receive results through an [`AioReadFuture`].
//!
//! ## Buffer ownership and drop-safety
//! The kernel writes into a destination buffer asynchronously, so that buffer
//! must stay valid until its completion is reaped — even if the awaiting future
//! is dropped early. Each read owns its buffer inside an [`IoSlot`]
//! (`Box<[u8]>`, a stable heap address). The reactor holds an `Arc<IoSlot>` in
//! its in-flight map for the whole duration of the request, so the buffer
//! outlives the kernel write regardless of what the caller does with the
//! future. On shutdown the reactor drops the queue **first** (which cancels and
//! drains every outstanding kernel write) and only then releases the slots, so
//! no buffer is ever freed while the kernel might still be writing to it.
//!
//! ## Back-pressure
//! macOS POSIX AIO caps the number of simultaneously in-flight requests
//! (`AIO_MAX`), and a batched `lio_listio` can accept an arbitrary subset of a
//! batch (the accepted set is not guaranteed to be a prefix). To keep the
//! correlation unambiguous the reactor submits **one op at a time**: a submit
//! that reports zero accepted is treated as back-pressure and retried after the
//! next completion is reaped. This trades a little batching for a simple,
//! correct accounting of which request is in flight.
//!
//! ## Honest platform scope
//! This bridge is only as capable as [`AioQueue`]: it has a real backend solely
//! on Apple/BSD today (POSIX AIO). On every other platform [`IoReactor::new`]
//! returns [`IoError::Unsupported`] and spawns no thread. Both batched reads
//! ([`IoReactor::read`]) and batched writes ([`IoReactor::write`]) are
//! supported; other opcodes (e.g. `fsync`) are PLANNED.
//!
//! ## Caller contract
//! As with [`AioQueue::submit`], the caller must keep the file descriptor open
//! until the request's future resolves (or the reactor is dropped). The read
//! destination buffer and the write source buffer are both owned by the bridge
//! (an [`IoSlot`] kept alive until the completion is reaped), so buffer
//! lifetime is always handled correctly here regardless of what the caller does
//! with the future.

use alloc::collections::VecDeque;
use alloc::sync::Arc;
use core::fmt;
use core::future::Future;
use core::pin::Pin;
use core::sync::atomic::{AtomicU64, Ordering};
use core::task::{Context, Poll, Waker};
use core::time::Duration;
use std::collections::HashMap;
use std::os::fd::RawFd;
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::Mutex;
use std::thread::{self, JoinHandle};

use prism_platform::aio::{AioError, AioQueue, IoPriority, ReadOp, WriteOp};

/// How long the reactor blocks in a single reap before looping back to pick up
/// newly submitted or back-pressured requests. Bounds the latency with which a
/// fresh submission is noticed while other reads are already in flight.
const REAP_TIMEOUT: Duration = Duration::from_millis(50);

/// Errors surfaced by the async-I/O bridge.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum IoError {
    /// This platform has no real async-I/O backend (see [`AioQueue`]).
    Unsupported,
    /// The reactor thread is gone (the [`IoReactor`] was dropped, or failed to
    /// start) so the request can never complete.
    ReactorGone,
    /// The underlying platform queue reported an error.
    Platform(AioError),
}

impl fmt::Display for IoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported => f.write_str("async I/O is unsupported on this platform"),
            Self::ReactorGone => f.write_str("async I/O reactor is no longer running"),
            Self::Platform(err) => write!(f, "async I/O backend error: {err}"),
        }
    }
}

impl std::error::Error for IoError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Platform(err) => Some(err),
            Self::Unsupported | Self::ReactorGone => None,
        }
    }
}

/// Result alias for the async-I/O bridge.
pub type Result<T> = core::result::Result<T, IoError>;

/// The buffer and bookkeeping shared between an [`AioReadFuture`] and the
/// reactor thread. Holds no raw pointers, so it is `Send + Sync`.
struct IoSlot {
    inner: Mutex<SlotInner>,
}

/// Mutable half of an [`IoSlot`].
struct SlotInner {
    /// Destination buffer. Present until the future takes it on completion; its
    /// heap address is stable while the kernel writes into it.
    buf: Option<Box<[u8]>>,
    /// Set once the reactor has reaped (or failed) this request.
    result: Option<Result<usize>>,
    /// Waker of the awaiting future, fired on completion.
    waker: Option<Waker>,
}

/// Store a request's outcome and wake the awaiting future (if any).
fn complete_slot(slot: &IoSlot, result: Result<usize>) {
    let waker = {
        let mut inner = slot.inner.lock().unwrap();
        inner.result = Some(result);
        inner.waker.take()
    };
    if let Some(waker) = waker {
        waker.wake();
    }
}

/// Which POSIX AIO opcode a [`Submission`] carries. The kernel either fills the
/// slot buffer (`Read`) or drains it (`Write`); the reactor branches on this to
/// pick the matching [`AioQueue`] entry point.
#[derive(Clone, Copy, PartialEq, Eq)]
enum OpKind {
    /// Kernel reads from the file into the slot buffer.
    Read,
    /// Kernel writes the slot buffer out to the file.
    Write,
}

/// An I/O request handed to the reactor thread. Every field is `Send`.
struct Submission {
    fd: RawFd,
    offset: u64,
    len: usize,
    priority: IoPriority,
    user_data: u64,
    kind: OpKind,
    slot: Arc<IoSlot>,
}

/// A dedicated async-I/O reactor backed by [`prism_platform::aio::AioQueue`].
///
/// Owns one thread that is the queue's sole owner. Submit reads with
/// [`IoReactor::read`] and `.await` (or [`crate::TaskPool::block_on`]) the
/// returned [`AioReadFuture`]. Dropping the reactor shuts the thread down,
/// cancelling any outstanding kernel writes and failing any pending futures
/// with [`IoError::ReactorGone`].
pub struct IoReactor {
    tx: Option<Sender<Submission>>,
    handle: Option<JoinHandle<()>>,
    next_id: AtomicU64,
}

impl IoReactor {
    /// Start a reactor, or report why one cannot run on this platform.
    ///
    /// Returns [`IoError::Unsupported`] (spawning no thread) when the platform
    /// has no real async-I/O backend, and [`IoError::Platform`] if the backend
    /// exists but the queue could not be created.
    pub fn new() -> Result<Self> {
        if !AioQueue::supported() {
            return Err(IoError::Unsupported);
        }

        let (tx, rx) = mpsc::channel::<Submission>();
        let (init_tx, init_rx) = mpsc::channel::<core::result::Result<(), AioError>>();

        let handle = thread::Builder::new()
            .name("prism-io-reactor".to_string())
            .spawn(move || match AioQueue::new() {
                Ok(queue) => {
                    // Report success; if the creator already gave up, just exit.
                    if init_tx.send(Ok(())).is_err() {
                        return;
                    }
                    drop(init_tx);
                    reactor_loop(queue, rx);
                }
                Err(err) => {
                    let _ = init_tx.send(Err(err));
                }
            })
            .map_err(|_| IoError::ReactorGone)?;

        match init_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                tx: Some(tx),
                handle: Some(handle),
                next_id: AtomicU64::new(0),
            }),
            Ok(Err(err)) => {
                let _ = handle.join();
                Err(IoError::Platform(err))
            }
            Err(_) => {
                let _ = handle.join();
                Err(IoError::ReactorGone)
            }
        }
    }

    /// Whether this build has a real async-I/O backend.
    #[must_use]
    pub fn supported() -> bool {
        AioQueue::supported()
    }

    /// Queue a read of `len` bytes from `fd` at `offset`, returning a future for
    /// the filled buffer. The read is submitted immediately; the future only
    /// observes its completion.
    ///
    /// The destination buffer is allocated by the bridge and returned inside the
    /// resolved [`ReadBuf`]. The caller must keep `fd` open until the future
    /// resolves.
    pub fn read(&self, fd: RawFd, offset: u64, len: usize, priority: IoPriority) -> AioReadFuture {
        self.read_into(
            fd,
            offset,
            priority,
            alloc::vec![0u8; len].into_boxed_slice(),
        )
    }

    /// Like [`IoReactor::read`], but reads into a caller-provided buffer (reused
    /// across reads to avoid reallocation). The number of bytes to read equals
    /// `buf.len()`.
    pub fn read_into(
        &self,
        fd: RawFd,
        offset: u64,
        priority: IoPriority,
        buf: Box<[u8]>,
    ) -> AioReadFuture {
        let len = buf.len();
        let user_data = self.next_id.fetch_add(1, Ordering::Relaxed);
        let slot = Arc::new(IoSlot {
            inner: Mutex::new(SlotInner {
                buf: Some(buf),
                result: None,
                waker: None,
            }),
        });

        let submission = Submission {
            fd,
            offset,
            len,
            priority,
            user_data,
            kind: OpKind::Read,
            slot: Arc::clone(&slot),
        };

        match &self.tx {
            Some(tx) if tx.send(submission).is_ok() => {}
            // The reactor thread has exited: resolve immediately with an error
            // (the un-sent `Submission` is dropped, releasing its slot clone).
            _ => complete_slot(&slot, Err(IoError::ReactorGone)),
        }

        AioReadFuture { slot, done: false }
    }

    /// Queue a write of `data` to `fd` at `offset`, returning a future that
    /// resolves to the number of bytes the kernel actually drained.
    ///
    /// The source bytes are copied into a bridge-owned buffer, so the caller may
    /// reuse or drop `data` immediately. The caller must keep `fd` open until
    /// the future resolves.
    pub fn write(
        &self,
        fd: RawFd,
        offset: u64,
        data: &[u8],
        priority: IoPriority,
    ) -> AioWriteFuture {
        self.write_from(fd, offset, priority, data.to_vec().into_boxed_slice())
    }

    /// Like [`IoReactor::write`], but takes ownership of `buf` as the source
    /// without copying. The number of bytes written equals `buf.len()`. `buf` is
    /// held by the bridge until the completion is reaped and then dropped.
    pub fn write_from(
        &self,
        fd: RawFd,
        offset: u64,
        priority: IoPriority,
        buf: Box<[u8]>,
    ) -> AioWriteFuture {
        let len = buf.len();
        let user_data = self.next_id.fetch_add(1, Ordering::Relaxed);
        let slot = Arc::new(IoSlot {
            inner: Mutex::new(SlotInner {
                buf: Some(buf),
                result: None,
                waker: None,
            }),
        });

        let submission = Submission {
            fd,
            offset,
            len,
            priority,
            user_data,
            kind: OpKind::Write,
            slot: Arc::clone(&slot),
        };

        match &self.tx {
            Some(tx) if tx.send(submission).is_ok() => {}
            // The reactor thread has exited: resolve immediately with an error
            // (the un-sent `Submission` is dropped, releasing its slot clone).
            _ => complete_slot(&slot, Err(IoError::ReactorGone)),
        }

        AioWriteFuture { slot, done: false }
    }
}

impl Drop for IoReactor {
    fn drop(&mut self) {
        // Disconnect the channel so the reactor sees `Disconnected`, drains,
        // drops its queue (cancelling in-flight kernel writes), fails any
        // outstanding slots, and exits. Then join it.
        self.tx.take();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// A filled read buffer.
///
/// `data` is the full destination buffer; `bytes` is how many leading bytes the
/// kernel actually transferred (which can be fewer than requested at end of
/// file). Read the valid region as `&buf.data[..buf.bytes]`.
#[derive(Debug)]
pub struct ReadBuf {
    /// The destination buffer (owned back by the caller).
    pub data: Box<[u8]>,
    /// Number of valid bytes transferred into `data`.
    pub bytes: usize,
}

impl ReadBuf {
    /// The slice of bytes actually read.
    #[must_use]
    pub fn filled(&self) -> &[u8] {
        &self.data[..self.bytes]
    }
}

/// The future returned by [`IoReactor::read`] / [`IoReactor::read_into`].
///
/// Resolves to the filled [`ReadBuf`] on success. Dropping it before completion
/// is safe: the reactor keeps the destination buffer alive until the kernel
/// write is reaped.
#[must_use = "an AioReadFuture does nothing unless awaited or blocked on"]
pub struct AioReadFuture {
    slot: Arc<IoSlot>,
    done: bool,
}

impl Future for AioReadFuture {
    type Output = Result<ReadBuf>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.done {
            // The single result was already yielded by an earlier poll.
            return Poll::Pending;
        }
        let mut inner = self.slot.inner.lock().unwrap();
        if let Some(result) = inner.result.take() {
            let output = match result {
                Ok(bytes) => {
                    let data = inner
                        .buf
                        .take()
                        .expect("completed read still owns its buffer");
                    Ok(ReadBuf { data, bytes })
                }
                Err(err) => Err(err),
            };
            drop(inner);
            self.done = true;
            return Poll::Ready(output);
        }
        match &mut inner.waker {
            Some(existing) if existing.will_wake(cx.waker()) => {}
            slot => *slot = Some(cx.waker().clone()),
        }
        Poll::Pending
    }
}

/// The future returned by [`IoReactor::write`] / [`IoReactor::write_from`].
///
/// Resolves to the number of bytes the kernel drained from the source buffer on
/// success. Dropping it before completion is safe: the reactor keeps the source
/// buffer alive until the kernel write is reaped.
#[must_use = "an AioWriteFuture does nothing unless awaited or blocked on"]
pub struct AioWriteFuture {
    slot: Arc<IoSlot>,
    done: bool,
}

impl Future for AioWriteFuture {
    type Output = Result<usize>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.done {
            // The single result was already yielded by an earlier poll.
            return Poll::Pending;
        }
        let mut inner = self.slot.inner.lock().unwrap();
        if let Some(result) = inner.result.take() {
            // The kernel is done with the source buffer; release it now.
            inner.buf.take();
            drop(inner);
            self.done = true;
            return Poll::Ready(result);
        }
        match &mut inner.waker {
            Some(existing) if existing.will_wake(cx.waker()) => {}
            slot => *slot = Some(cx.waker().clone()),
        }
        Poll::Pending
    }
}

/// The reactor thread's body: own the queue, submit pending reads one at a time,
/// reap completions, and shut down cleanly when the channel disconnects.
fn reactor_loop(mut queue: AioQueue, rx: Receiver<Submission>) {
    let mut inflight: HashMap<u64, Arc<IoSlot>> = HashMap::new();
    let mut pending: VecDeque<Submission> = VecDeque::new();
    let mut disconnected = false;
    let mut fatal: Option<IoError> = None;

    'main: loop {
        // 1. Drain the channel without blocking.
        loop {
            match rx.try_recv() {
                Ok(sub) => pending.push_back(sub),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    disconnected = true;
                    break;
                }
            }
        }

        // 2. Submit queued requests one at a time, honoring back-pressure.
        while let Some(sub) = pending.front() {
            // The slot buffer has a stable heap address; capture its pointer
            // under the lock, then submit without holding it.
            let ptr = {
                let mut inner = sub.slot.inner.lock().unwrap();
                inner
                    .buf
                    .as_mut()
                    .expect("submission buffer present before submit")
                    .as_mut_ptr()
            };
            // SAFETY (both arms): `ptr` points into the slot's `Box<[u8]>`,
            // which stays allocated (and at a stable address) because the
            // reactor holds an `Arc<IoSlot>` in `inflight` until the completion
            // is reaped; the caller's contract keeps `fd` open for the same
            // window. For a write the source bytes are owned by the slot and are
            // never mutated while in flight.
            let accepted = match sub.kind {
                OpKind::Read => {
                    let op = ReadOp {
                        fd: sub.fd,
                        offset: sub.offset,
                        buf: ptr,
                        len: sub.len,
                        user_data: sub.user_data,
                        priority: sub.priority,
                    };
                    #[expect(unsafe_code, reason = "forward one read op to the platform aio queue")]
                    // SAFETY: see the block comment above; `ptr` is the slot's
                    // stable buffer, kept alive via `inflight`, and `fd` stays
                    // open per the caller's contract.
                    unsafe {
                        queue.submit(core::slice::from_ref(&op))
                    }
                }
                OpKind::Write => {
                    let op = WriteOp {
                        fd: sub.fd,
                        offset: sub.offset,
                        buf: ptr.cast_const(),
                        len: sub.len,
                        user_data: sub.user_data,
                        priority: sub.priority,
                    };
                    #[expect(
                        unsafe_code,
                        reason = "forward one write op to the platform aio queue"
                    )]
                    // SAFETY: see the block comment above; `ptr` is the slot's
                    // stable source buffer, kept alive via `inflight` and never
                    // mutated in flight, and `fd` stays open per the caller's
                    // contract.
                    unsafe {
                        queue.submit_write(core::slice::from_ref(&op))
                    }
                }
            };
            match accepted {
                Ok(0) => break, // kernel back-pressure; retry after a reap
                Ok(_) => {
                    let sub = pending.pop_front().expect("front checked above");
                    inflight.insert(sub.user_data, Arc::clone(&sub.slot));
                }
                Err(err) => {
                    let sub = pending.pop_front().expect("front checked above");
                    complete_slot(&sub.slot, Err(IoError::Platform(err)));
                }
            }
        }

        // 3. With nothing in flight, block for the next submission (or exit).
        if inflight.is_empty() {
            if pending.is_empty() {
                if disconnected {
                    break 'main;
                }
                match rx.recv() {
                    Ok(sub) => pending.push_back(sub),
                    Err(_) => break 'main,
                }
            } else {
                // Pending reads are all back-pressured with nothing in flight to
                // reap; yield and retry rather than busy-spinning.
                thread::yield_now();
            }
            continue;
        }

        // 4. Reap completions (bounded so new submissions are noticed promptly).
        match queue.wait(usize::MAX, Some(REAP_TIMEOUT)) {
            Ok(completions) => {
                for completion in completions {
                    if let Some(slot) = inflight.remove(&completion.user_data) {
                        complete_slot(&slot, completion.result.map_err(IoError::Platform));
                    }
                }
            }
            Err(err) => {
                fatal = Some(IoError::Platform(err));
                break 'main;
            }
        }
    }

    // Shutdown. Drop the queue first so the kernel cancels and drains every
    // outstanding request while the slot buffers (kept alive by the slot Arcs
    // in `inflight`) are still valid; only then fail the waiting futures.
    drop(queue);
    let shutdown_err = fatal.unwrap_or(IoError::ReactorGone);
    for (_, slot) in inflight.drain() {
        complete_slot(&slot, Err(shutdown_err.clone()));
    }
    for sub in pending.drain(..) {
        complete_slot(&sub.slot, Err(shutdown_err.clone()));
    }
}

#[cfg(test)]
mod tests;
