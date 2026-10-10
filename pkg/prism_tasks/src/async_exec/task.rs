//! [`Task<T>`]: the join handle for a spawned future.
//!
//! A `Task<T>` shares a [`ResultCell`] with the running future. The future
//! stores its output into the cell on completion; the handle can then be
//! `.await`ed, polled, or drained synchronously via
//! [`TaskPool::block_on`](crate::TaskPool::block_on). Each task also owns a
//! completion [`Counter`] that drains when it finishes, so a *job* can wait on
//! an async result with [`TaskPool::wait`](crate::TaskPool::wait) — the same
//! bridge the async side uses via [`Counter::wait_async`](crate::Counter::wait_async).

use alloc::sync::Arc;
use std::future::Future;
use std::pin::Pin;
use std::sync::Mutex;
use std::task::{Context, Poll, Waker};

use crate::Counter;

/// Shared slot connecting a spawned future to its [`Task`] handle.
pub(crate) struct ResultCell<T> {
    state: Mutex<CellState<T>>,
    /// Completes (`1 -> 0`) when the future finishes; lets jobs wait on the
    /// task and lets [`Counter::wait_async`](crate::Counter::wait_async) bridge
    /// task completion back into the async world.
    counter: Counter,
}

/// The mutable half of a [`ResultCell`].
struct CellState<T> {
    /// The produced value, moved out by the first successful poll/await.
    value: Option<T>,
    /// Set once the future has finished (distinguishes "no value yet" from
    /// "value already taken").
    done: bool,
    /// Waker of a handle awaiting the result, fired on completion.
    waker: Option<Waker>,
}

impl<T> ResultCell<T> {
    /// Create an empty cell with its completion counter armed at one.
    pub(crate) fn new() -> Arc<Self> {
        let counter = Counter::new();
        counter.add(1);
        Arc::new(Self {
            state: Mutex::new(CellState {
                value: None,
                done: false,
                waker: None,
            }),
            counter,
        })
    }

    /// Store the future's output and wake any waiting handle and the counter.
    pub(crate) fn complete(&self, value: T) {
        let waker = {
            let mut state = self.state.lock().unwrap();
            state.value = Some(value);
            state.done = true;
            state.waker.take()
        };
        // Drain the completion counter first (wakes job-side / counter waiters),
        // then the handle's own waker.
        self.counter.finish_one();
        if let Some(waker) = waker {
            waker.wake();
        }
    }

    /// Poll for the result, parking `waker` if it is not ready yet.
    fn poll_result(&self, waker: &Waker) -> Poll<T> {
        let mut state = self.state.lock().unwrap();
        if let Some(value) = state.value.take() {
            return Poll::Ready(value);
        }
        if state.done {
            // Completed, but the value was already taken by an earlier poll.
            // A `Task` is single-consumer, so this only happens if a handle is
            // polled after it already yielded `Ready`; keep returning pending
            // rather than panicking across the await boundary.
            return Poll::Pending;
        }
        match &mut state.waker {
            Some(existing) if existing.will_wake(waker) => {}
            slot => *slot = Some(waker.clone()),
        }
        Poll::Pending
    }

    fn is_finished(&self) -> bool {
        self.state.lock().unwrap().done
    }
}

/// A handle to a spawned future's eventual output.
///
/// Obtained from [`TaskPool::spawn_async`](crate::TaskPool::spawn_async). The
/// handle implements [`Future`], so it can be `.await`ed from another async
/// task, and can be drained synchronously with
/// [`TaskPool::block_on`](crate::TaskPool::block_on). Dropping a `Task`
/// *detaches* it: the underlying future keeps running to completion.
#[must_use = "a Task does nothing unless awaited, blocked on, or explicitly detached"]
pub struct Task<T> {
    cell: Arc<ResultCell<T>>,
}

impl<T> Task<T> {
    pub(crate) fn new(cell: Arc<ResultCell<T>>) -> Self {
        Self { cell }
    }

    /// Whether the spawned future has finished (its result may already have
    /// been consumed).
    pub fn is_finished(&self) -> bool {
        self.cell.is_finished()
    }

    /// A [`Counter`] that drains when this task completes.
    ///
    /// Lets a *job* wait on the task with
    /// [`TaskPool::wait`](crate::TaskPool::wait), bridging the async result back
    /// into the fork-join world.
    pub fn counter(&self) -> Counter {
        self.cell.counter.clone()
    }

    /// Detach the task, letting its future run to completion in the background.
    ///
    /// This is exactly what dropping the handle does; the method documents the
    /// intent at the call site.
    pub fn detach(self) {
        drop(self);
    }
}

impl<T> Future for Task<T> {
    type Output = T;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<T> {
        self.cell.poll_result(cx.waker())
    }
}
