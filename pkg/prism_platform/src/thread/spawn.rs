//! Thread spawning and basic thread controls over [`std::thread`].
//!
//! [`Builder`] configures an optional name and stack size and produces a typed
//! [`JoinHandle`]. The free [`spawn`] function is a convenience for the common
//! "spawn and join a value" case. The remaining helpers forward to `std` and
//! reuse [`crate::cpu`] for the logical-core count.

use std::time::Duration;

/// Re-export of the opaque per-thread identifier from [`std::thread`].
pub use std::thread::ThreadId;

/// A handle that owns the join rendezvous for a spawned thread.
///
/// Thin wrapper over [`std::thread::JoinHandle`] so the engine depends on a
/// single platform surface instead of `std::thread` directly.
#[derive(Debug)]
pub struct JoinHandle<T> {
    inner: std::thread::JoinHandle<T>,
}

impl<T> JoinHandle<T> {
    /// Wait for the associated thread to finish and return its produced value.
    ///
    /// Returns [`Err`] carrying the panic payload if the thread panicked,
    /// matching [`std::thread::JoinHandle::join`].
    pub fn join(self) -> std::thread::Result<T> {
        self.inner.join()
    }

    /// Returns `true` once the associated thread has finished running.
    pub fn is_finished(&self) -> bool {
        self.inner.is_finished()
    }

    /// Borrow the underlying [`std::thread::Thread`] handle (for `unpark`,
    /// name lookup, and identity).
    pub fn thread(&self) -> &std::thread::Thread {
        self.inner.thread()
    }
}

/// Builder for a configured thread: optional name and stack size.
#[derive(Clone, Debug, Default)]
pub struct Builder {
    name: Option<String>,
    stack_size: Option<usize>,
}

impl Builder {
    /// Create a builder with no name and the platform-default stack size.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the OS/debugger-visible thread name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the thread stack size in bytes.
    pub fn stack_size(mut self, bytes: usize) -> Self {
        self.stack_size = Some(bytes);
        self
    }

    /// Spawn the thread, running `f` and yielding a [`JoinHandle`].
    ///
    /// Returns an [`std::io::Error`] if the OS refuses to create the thread.
    pub fn spawn<F, T>(self, f: F) -> std::io::Result<JoinHandle<T>>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        let mut builder = std::thread::Builder::new();
        if let Some(name) = self.name {
            builder = builder.name(name);
        }
        if let Some(bytes) = self.stack_size {
            builder = builder.stack_size(bytes);
        }
        builder.spawn(f).map(|inner| JoinHandle { inner })
    }
}

/// Spawn an unnamed thread with the default stack size.
///
/// Convenience over [`Builder::spawn`]; panics if the OS cannot create the
/// thread, matching [`std::thread::spawn`]. Use [`Builder`] when a name, a
/// custom stack size, or fallible creation is required.
pub fn spawn<F, T>(f: F) -> JoinHandle<T>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    JoinHandle {
        inner: std::thread::spawn(f),
    }
}

/// Yield the current timeslice back to the OS scheduler.
pub fn yield_now() {
    std::thread::yield_now();
}

/// Block the current thread for at least `dur`.
pub fn sleep(dur: Duration) {
    std::thread::sleep(dur);
}

/// Identifier of the calling thread.
pub fn current_id() -> ThreadId {
    std::thread::current().id()
}

/// Number of logical cores usable for thread-pool sizing.
///
/// Reuses [`crate::cpu::CpuInfo`] so callers share one topology probe; always
/// returns at least `1`.
pub fn hardware_concurrency() -> usize {
    crate::cpu::CpuInfo::detect().logical_cores.max(1)
}
