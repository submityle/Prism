//! Token-based thread parking.
//!
//! [`Parker`]/[`Unparker`] wrap [`std::thread::park`] and
//! [`std::thread::Thread::unpark`] into an explicit pair with a single wakeup
//! token. The token makes the handshake race-free: an [`Unparker::unpark`]
//! that happens *before* the matching [`Parker::park`] is remembered, so the
//! subsequent `park` returns immediately instead of blocking forever. Spurious
//! wakeups from the OS are absorbed internally, so `park` only returns once a
//! real token is present.
//!
//! A [`Parker`] is bound at construction to the thread that creates it; that
//! same thread must be the one that calls [`Parker::park`]. The paired
//! [`Unparker`] is cloneable and `Send`, so any thread may wake the parked one.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use std::thread::Thread;
use std::time::Duration;

/// No token present and no one parked.
const EMPTY: u8 = 0;
/// The owning thread is (about to be) parked, waiting for a token.
const PARKED: u8 = 1;
/// A token has been posted; the next `park` consumes it without blocking.
const NOTIFIED: u8 = 2;

/// Shared state behind a [`Parker`]/[`Unparker`] pair.
#[derive(Debug)]
struct Inner {
    state: AtomicU8,
    owner: Thread,
}

/// The parking side of the pair; owned and used by a single thread.
#[derive(Debug)]
pub struct Parker {
    inner: Arc<Inner>,
}

/// The waking side of the pair; cloneable and shareable across threads.
#[derive(Clone, Debug)]
pub struct Unparker {
    inner: Arc<Inner>,
}

impl Parker {
    /// Create a parker bound to the calling thread.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Inner {
                state: AtomicU8::new(EMPTY),
                owner: std::thread::current(),
            }),
        }
    }

    /// Obtain an [`Unparker`] that can wake this parker from any thread.
    pub fn unparker(&self) -> Unparker {
        Unparker {
            inner: Arc::clone(&self.inner),
        }
    }

    /// Block the calling thread until a token is available, then consume it.
    ///
    /// Must be called by the thread that created this [`Parker`]. If a token
    /// was already posted, returns immediately.
    pub fn park(&self) {
        // Fast path: a token is already waiting.
        if self
            .inner
            .state
            .compare_exchange(NOTIFIED, EMPTY, Ordering::Acquire, Ordering::Acquire)
            .is_ok()
        {
            return;
        }
        // Announce that we are about to park.
        match self.inner.state.compare_exchange(
            EMPTY,
            PARKED,
            Ordering::Acquire,
            Ordering::Acquire,
        ) {
            Ok(_) => {}
            // A token arrived between the two checks: consume and return.
            Err(NOTIFIED) => {
                self.inner.state.store(EMPTY, Ordering::Release);
                return;
            }
            Err(_) => {}
        }
        loop {
            std::thread::park();
            // Only a real token (NOTIFIED) ends the wait; otherwise re-park to
            // absorb spurious wakeups.
            if self
                .inner
                .state
                .compare_exchange(NOTIFIED, EMPTY, Ordering::Acquire, Ordering::Acquire)
                .is_ok()
            {
                return;
            }
        }
    }

    /// Like [`Parker::park`] but gives up after at most `timeout`.
    ///
    /// Returns `true` if a token was consumed, `false` if the timeout elapsed
    /// first. May return early on a spurious wakeup without a token, reporting
    /// `false`.
    pub fn park_timeout(&self, timeout: Duration) -> bool {
        if self
            .inner
            .state
            .compare_exchange(NOTIFIED, EMPTY, Ordering::Acquire, Ordering::Acquire)
            .is_ok()
        {
            return true;
        }
        match self.inner.state.compare_exchange(
            EMPTY,
            PARKED,
            Ordering::Acquire,
            Ordering::Acquire,
        ) {
            Ok(_) => {}
            Err(NOTIFIED) => {
                self.inner.state.store(EMPTY, Ordering::Release);
                return true;
            }
            Err(_) => {}
        }
        std::thread::park_timeout(timeout);
        // Reclaim the slot regardless of why we woke; report whether a token
        // was actually posted.
        self.inner.state.swap(EMPTY, Ordering::Acquire) == NOTIFIED
    }
}

impl Default for Parker {
    fn default() -> Self {
        Self::new()
    }
}

impl Unparker {
    /// Post a wakeup token, unblocking the parked owner (or arranging for its
    /// next [`Parker::park`] to return immediately). Idempotent until consumed.
    pub fn unpark(&self) {
        // Publish the token; wake the owner only if it was actually parked.
        if self.inner.state.swap(NOTIFIED, Ordering::Release) == PARKED {
            self.inner.owner.unpark();
        }
    }
}
