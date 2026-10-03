//! Fork-join dependency counter.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// An atomic countdown used to track outstanding jobs in a fork-join group.
///
/// [`TaskPool::spawn`](crate::TaskPool::spawn) increments the counter and the
/// worker decrements it when the job finishes. [`TaskPool::wait`] blocks (while
/// helping run other jobs) until the counter reaches zero.
#[derive(Clone, Default)]
pub struct Counter {
    inner: Arc<AtomicUsize>,
}

impl Counter {
    /// Create a new counter at zero.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Number of jobs still outstanding.
    pub fn pending(&self) -> usize {
        self.inner.load(Ordering::Acquire)
    }

    /// Whether all tracked jobs have completed.
    pub fn is_complete(&self) -> bool {
        self.pending() == 0
    }

    pub(crate) fn add(&self, n: usize) {
        self.inner.fetch_add(n, Ordering::Release);
    }

    pub(crate) fn finish_one(&self) {
        self.inner.fetch_sub(1, Ordering::Release);
    }
}
