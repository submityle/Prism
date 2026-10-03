//! The [`Counter`] <-> future bridge.
//!
//! [`Counter::wait_async`] yields a [`CounterFuture`] that completes when the
//! counter drains to zero, letting an async task `.await` the completion of a
//! fork-join group. The reverse direction — a job waiting on an async result —
//! is provided by [`Task::counter`](crate::Task::counter), whose counter this
//! same machinery drains on task completion.

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use crate::Counter;

/// A future that resolves once its [`Counter`] has reached zero.
///
/// Created by [`Counter::wait_async`]. Polling registers the task's waker with
/// the counter; the counter's `1 -> 0` transition fires it. The registration
/// and the completion are serialized under the counter's waker lock, so a
/// completion that races the first poll is never missed.
#[must_use = "a CounterFuture does nothing unless awaited"]
pub struct CounterFuture {
    counter: Counter,
}

impl CounterFuture {
    pub(crate) fn new(counter: Counter) -> Self {
        Self { counter }
    }
}

impl Future for CounterFuture {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if self.counter.is_complete() {
            return Poll::Ready(());
        }
        // `register_waker` re-checks completion under the drain lock and
        // reports `true` if it already fired, closing the register/complete gap.
        if self.counter.register_waker(cx.waker()) {
            Poll::Ready(())
        } else {
            Poll::Pending
        }
    }
}

impl Counter {
    /// A future that completes when this counter next reaches zero.
    ///
    /// Bridges the fork-join world into async code: `counter.wait_async().await`
    /// suspends the awaiting task until every job tracked by `counter` finishes.
    pub fn wait_async(&self) -> CounterFuture {
        CounterFuture::new(self.clone())
    }
}
