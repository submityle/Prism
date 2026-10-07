//! §24.1 Advanced asynchronous I/O: a submission / completion queue facade for
//! high-throughput streaming reads.
//!
//! Open-world AAA streaming cannot feed the GPU with synchronous `read`s: it
//! needs to hand the kernel a *batch* of read requests in a single syscall and
//! later reap their completions out of order. This module exposes exactly that
//! shape — [`AioQueue::submit`] enqueues N [`ReadOp`]s (and [`AioQueue::submit_write`]
//! N [`WriteOp`]s) at once and [`AioQueue::wait`] drains the ready
//! [`Completion`]s — on top of whichever real OS primitive the target platform
//! offers.
//!
//! ## Backends
//! - **Apple (macOS/iOS) & BSD**: real POSIX AIO (`lio_listio` for one-syscall
//!   batch submit, `aio_suspend`/`aio_error`/`aio_return` to reap, `aio_cancel`
//!   on drop). This is the backend exercised by this crate's test-suite on the
//!   Apple-silicon CI host. `kqueue` is already owned by the file watcher, so
//!   `lio_listio` is the honest macOS realization of "batch submit N reads".
//! - **Every other OS** (Linux `io_uring`, Windows IOCP, Web): honestly
//!   returns [`AioError::Unsupported`] from [`AioQueue::new`]. Those real
//!   backends are PLANNED; this crate ships no stub that pretends to read.
//!   Query [`AioQueue::supported`] first.
//!
//! ## Safety contract
//! [`AioQueue::submit`] / [`AioQueue::submit_write`] are `unsafe`: each
//! [`ReadOp`] / [`WriteOp`] carries a raw `fd`, a `buf` pointer, and a `len`.
//! The caller must keep the file descriptor open and the `buf`/`len` region
//! valid until the matching [`Completion`] has been reaped by
//! [`AioQueue::wait`] (or the queue has been dropped, which cancels and drains
//! every outstanding request before freeing its control blocks). For a read the
//! kernel *writes into* the destination `buf`, so it must be exclusively
//! borrowed; for a write the kernel *reads from* the source `buf`, so it must
//! stay valid **and unmodified** for the same window. Releasing or mutating the
//! region early is undefined behavior.

use core::fmt;
use core::time::Duration;
use std::os::fd::RawFd;

#[cfg(all(
    feature = "std",
    any(
        target_vendor = "apple",
        target_os = "freebsd",
        target_os = "netbsd",
        target_os = "dragonfly"
    )
))]
#[path = "posix.rs"]
mod backend;

#[cfg(all(
    feature = "std",
    not(any(
        target_vendor = "apple",
        target_os = "freebsd",
        target_os = "netbsd",
        target_os = "dragonfly"
    ))
))]
mod backend {
    //! Fallback backend for platforms without a batch async-I/O primitive wired
    //! up yet (Linux `io_uring`, Windows IOCP, Web). Every entry point honestly
    //! reports [`super::AioError::Unsupported`] rather than faking a read.

    use super::{AioError, Completion, ReadOp, Result, WriteOp};
    use core::time::Duration;

    /// This build has no real async-I/O backend.
    pub(super) const SUPPORTED: bool = false;

    /// Placeholder queue; never constructed because [`Queue::new`] fails.
    pub(super) struct Queue;

    impl Queue {
        /// Construction is unsupported here.
        pub(super) fn new() -> Result<Self> {
            Err(AioError::Unsupported)
        }

        /// Submit is unreachable (no instance can exist); honest `Unsupported`.
        ///
        /// # Safety
        /// Unreachable: no `Queue` can be constructed on this platform.
        pub(super) unsafe fn submit(&mut self, _ops: &[ReadOp]) -> Result<usize> {
            Err(AioError::Unsupported)
        }

        /// Write submit is unreachable (no instance can exist); honest
        /// `Unsupported`.
        ///
        /// # Safety
        /// Unreachable: no `Queue` can be constructed on this platform.
        pub(super) unsafe fn submit_write(&mut self, _ops: &[WriteOp]) -> Result<usize> {
            Err(AioError::Unsupported)
        }

        /// Wait is unreachable (no instance can exist); honest `Unsupported`.
        pub(super) fn wait(
            &mut self,
            _max: usize,
            _timeout: Option<Duration>,
        ) -> Result<Vec<Completion>> {
            Err(AioError::Unsupported)
        }

        /// No request can be in flight here.
        pub(super) fn pending(&self) -> usize {
            0
        }
    }
}

/// Errors surfaced by the async-I/O facade.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AioError {
    /// This platform has no real batch async-I/O backend wired up.
    Unsupported,
    /// A [`ReadOp`] was malformed (e.g. a negative file descriptor).
    InvalidArgument,
    /// An OS error occurred; the payload is the raw `errno`.
    Io(i32),
}

impl fmt::Display for AioError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported => f.write_str("async I/O is unsupported on this platform"),
            Self::InvalidArgument => f.write_str("invalid async-I/O request"),
            Self::Io(code) => write!(f, "async I/O OS error (errno {code})"),
        }
    }
}

impl std::error::Error for AioError {}

/// Result alias for the async-I/O facade.
pub type Result<T> = core::result::Result<T, AioError>;

/// Relative scheduling hint for a read request.
///
/// Maps to the POSIX `aio_reqprio` delta on backends that honor it. Many
/// kernels (notably macOS) largely ignore it, so this is strictly best-effort;
/// correctness never depends on the ordering it requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IoPriority {
    /// Latency-critical streaming (e.g. geometry in front of the camera).
    High,
    /// Ordinary streaming workload.
    #[default]
    Normal,
    /// Opportunistic prefetch that must yield to everything else.
    Low,
}

/// A single batched read request.
///
/// `buf`/`len` describe the destination the kernel fills asynchronously; see
/// the module-level safety contract. `user_data` is an opaque token echoed back
/// on the matching [`Completion`] so callers can correlate out-of-order
/// completions with their requests.
#[derive(Debug, Clone, Copy)]
pub struct ReadOp {
    /// Source file descriptor (must stay open until reaped).
    pub fd: RawFd,
    /// Byte offset within the file to read from.
    pub offset: u64,
    /// Destination buffer (must stay valid until reaped).
    pub buf: *mut u8,
    /// Number of bytes to read.
    pub len: usize,
    /// Opaque correlation token echoed on the [`Completion`].
    pub user_data: u64,
    /// Best-effort scheduling hint.
    pub priority: IoPriority,
}

/// A single batched write request.
///
/// `buf`/`len` describe the *source* bytes the kernel drains asynchronously;
/// unlike [`ReadOp`] the buffer is read-only to the kernel, so `buf` is a
/// `*const u8`. The caller must keep that source region valid and unmodified
/// until the matching [`Completion`] is reaped (see the module-level contract).
/// `user_data` is an opaque token echoed back on the [`Completion`].
#[derive(Debug, Clone, Copy)]
pub struct WriteOp {
    /// Destination file descriptor (must stay open until reaped).
    pub fd: RawFd,
    /// Byte offset within the file to write to.
    pub offset: u64,
    /// Source buffer (must stay valid and unmodified until reaped).
    pub buf: *const u8,
    /// Number of bytes to write.
    pub len: usize,
    /// Opaque correlation token echoed on the [`Completion`].
    pub user_data: u64,
    /// Best-effort scheduling hint.
    pub priority: IoPriority,
}

/// A reaped async-read result.
#[derive(Debug, Clone)]
pub struct Completion {
    /// The [`ReadOp::user_data`] token of the request this completes.
    pub user_data: u64,
    /// On success, the number of bytes transferred; otherwise the OS error.
    pub result: Result<usize>,
}

/// A batch async-read submission/completion queue.
///
/// Construct with [`AioQueue::new`] (fails with [`AioError::Unsupported`] where
/// no real backend exists), submit batches with [`AioQueue::submit`], and reap
/// completions with [`AioQueue::wait`]. Dropping the queue cancels and drains
/// every outstanding request before freeing its kernel control blocks.
pub struct AioQueue {
    inner: backend::Queue,
}

impl AioQueue {
    /// Create a new async-I/O queue, or [`AioError::Unsupported`] if this
    /// platform has no real backend.
    pub fn new() -> Result<Self> {
        Ok(Self {
            inner: backend::Queue::new()?,
        })
    }

    /// Whether this build has a real async-I/O backend.
    #[must_use]
    pub fn supported() -> bool {
        backend::SUPPORTED
    }

    /// Submit a batch of reads in as few syscalls as the backend allows.
    ///
    /// Returns the number of requests accepted into the in-flight set (normally
    /// `ops.len()`); fewer indicates the kernel applied back-pressure and the
    /// caller should retry the tail later. Empty input is a no-op (`Ok(0)`).
    ///
    /// # Safety
    /// Every [`ReadOp`]'s `fd` must stay open and its `buf`/`len` region must
    /// stay valid and exclusively borrowed until the matching [`Completion`] is
    /// reaped (or this queue is dropped). See the module-level contract.
    #[expect(
        unsafe_code,
        reason = "forwarding the unsafe batch-submit to the platform backend"
    )]
    pub unsafe fn submit(&mut self, ops: &[ReadOp]) -> Result<usize> {
        // SAFETY: forwarded to the backend under the caller's module-level
        // promise that each op's fd and buffer outlive its completion.
        unsafe { self.inner.submit(ops) }
    }

    /// Submit a batch of writes in as few syscalls as the backend allows.
    ///
    /// Returns the number of requests accepted into the in-flight set (normally
    /// `ops.len()`); fewer indicates the kernel applied back-pressure and the
    /// caller should retry the tail later. Empty input is a no-op (`Ok(0)`).
    /// Completions are reaped by [`AioQueue::wait`] exactly like reads;
    /// correlate them via [`Completion::user_data`].
    ///
    /// # Safety
    /// Every [`WriteOp`]'s `fd` must stay open and its `buf`/`len` source region
    /// must stay valid and unmodified until the matching [`Completion`] is
    /// reaped (or this queue is dropped). See the module-level contract.
    #[expect(
        unsafe_code,
        reason = "forwarding the unsafe batch-submit to the platform backend"
    )]
    pub unsafe fn submit_write(&mut self, ops: &[WriteOp]) -> Result<usize> {
        // SAFETY: forwarded to the backend under the caller's module-level
        // promise that each op's fd and source buffer outlive its completion
        // and stay unmodified until it is reaped.
        unsafe { self.inner.submit_write(ops) }
    }

    /// Block until at least one request completes (or `timeout` elapses), then
    /// reap up to `max` ready completions.
    ///
    /// Returns an empty vector on timeout or when nothing is in flight. The
    /// order of returned completions is not the submission order; correlate via
    /// [`Completion::user_data`].
    pub fn wait(&mut self, max: usize, timeout: Option<Duration>) -> Result<Vec<Completion>> {
        self.inner.wait(max, timeout)
    }

    /// Number of requests currently in flight (submitted but not yet reaped).
    #[must_use]
    pub fn pending(&self) -> usize {
        self.inner.pending()
    }
}

#[cfg(test)]
mod tests;
