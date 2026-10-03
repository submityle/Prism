//! Fork-join dependency counter.

use alloc::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::Waker;

use std::sync::Mutex;

/// An atomic countdown used to track outstanding jobs in a fork-join group.
///
/// [`TaskPool::spawn`](crate::TaskPool::spawn) increments the counter and the
/// worker decrements it when the job finishes. [`TaskPool::wait`](crate::TaskPool::wait)
/// blocks (while helping run other jobs) until the counter reaches zero.
///
/// The counter also bridges into the async executor: [`Counter::wait_async`]
/// yields a [`CounterFuture`](crate::CounterFuture) that completes when the
/// counter drains, and the `1 -> 0` transition wakes any such awaiting futures.
#[derive(Clone, Default)]
pub struct Counter {
    inner: Arc<Inner>,
}

/// Shared counter state: the atomic count plus any async wakers parked on it.
#[derive(Default)]
struct Inner {
    /// Number of jobs still outstanding.
    count: AtomicUsize,
    /// Wakers of futures awaiting completion; drained and fired on `1 -> 0`.
    wakers: Mutex<Vec<Waker>>,
}

impl Counter {
    /// Create a new counter at zero.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Inner::default()),
        }
    }

    /// Number of jobs still outstanding.
    pub fn pending(&self) -> usize {
        self.inner.count.load(Ordering::Acquire)
    }

    /// Whether all tracked jobs have completed.
    pub fn is_complete(&self) -> bool {
        self.pending() == 0
    }

    pub(crate) fn add(&self, n: usize) {
        self.inner.count.fetch_add(n, Ordering::Release);
    }

    pub(crate) fn finish_one(&self) {
        // `AcqRel` so the completing thread both publishes its writes and
        // observes a concurrent `register_waker` that ran before this decrement.
        let prev = self.inner.count.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(prev > 0, "counter underflow: finish_one past zero");
        if prev == 1 {
            self.wake_all();
        }
    }

    /// Register `waker` to be notified when this counter next reaches zero.
    ///
    /// Returns `true` if the counter is *already* complete (the caller should
    /// treat the wait as resolved and not park). Checking completion under the
    /// same lock [`Counter::finish_one`] takes to drain closes the register/
    /// complete race: either this call observes the completion and returns
    /// `true`, or the waker is enqueued before the drain and is fired by it.
    pub(crate) fn register_waker(&self, waker: &Waker) -> bool {
        let mut wakers = self.inner.wakers.lock().unwrap();
        if self.is_complete() {
            return true;
        }
        if !wakers.iter().any(|w| w.will_wake(waker)) {
            wakers.push(waker.clone());
        }
        false
    }

    /// Drain and fire every parked waker. Wakers run outside the lock so a woken
    /// task re-polling (and re-registering) cannot deadlock on the waker mutex.
    fn wake_all(&self) {
        let drained: Vec<Waker> = {
            let mut wakers = self.inner.wakers.lock().unwrap();
            std::mem::take(&mut *wakers)
        };
        for waker in drained {
            waker.wake();
        }
    }
}
