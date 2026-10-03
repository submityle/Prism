//! The minimal future executor: it schedules futures as jobs on the existing
//! [`TaskPool`] and drives them with a hand-rolled waker.
//!
//! ## Model
//! Spawning a future wraps it in a non-generic [`TaskHarness`] (the typed
//! output is funneled into a [`ResultCell`] shared with the [`Task`] handle).
//! The harness is a [`WakeTask`]: when its waker fires it re-enqueues itself.
//! A multi-threaded pool runs each poll as an ordinary job on a worker (so a
//! future shares the work-stealing load balancer with the job graph, exactly as
//! the design intends); the single-threaded fallback keeps a ready queue that
//! [`TaskPool::block_on`] drains inline.
//!
//! ## Deadlock-freedom
//! A harness is enqueued at most once at a time (a small [`RunState`] machine
//! guards against double-enqueue and handles a wake that lands *during* a poll
//! by re-polling instead of re-queuing). [`TaskPool::block_on`] parks the
//! calling thread on a condvar and is woken by the future's own waker, so it
//! never busy-spins; in the single-threaded fallback it instead drives the
//! ready queue and fails fast if a future can make no progress, so it can never
//! hang.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Condvar, Mutex};
use std::task::{Context, Poll};

use super::task::{ResultCell, Task};
use super::waker::{WakeTask, waker_of};
use crate::TaskPool;
use crate::scheduler::Shared;

/// A type-erased, re-schedulable unit the scheduler can run without knowing the
/// future's output type.
pub(crate) trait RunnableTask: Send + Sync {
    /// Poll the underlying future once (driving the task forward).
    fn run_boxed(self: Arc<Self>);
}

/// Lifecycle of a harness, guarding against double-scheduling and wakes that
/// arrive while a poll is in flight.
#[derive(Clone, Copy, PartialEq, Eq)]
enum RunState {
    /// Not queued and not running.
    Idle,
    /// Sitting in a run queue, awaiting a poll.
    Queued,
    /// Currently being polled.
    Running,
    /// Woken while being polled; the runner must poll again.
    RunningNotified,
    /// Finished; the future has been dropped.
    Done,
}

/// A spawned future plus the state needed to re-schedule it.
struct TaskHarness {
    inner: Mutex<HarnessInner>,
    /// Pool state used to re-enqueue; shared with the owning [`TaskPool`].
    shared: Arc<Shared>,
    /// Whether the owning pool is the single-threaded fallback (no workers).
    single: bool,
}

/// The mutable half of a [`TaskHarness`].
struct HarnessInner {
    /// The future, taken out while being polled and dropped once `Ready`.
    future: Option<Pin<Box<dyn Future<Output = ()> + Send>>>,
    run: RunState,
}

impl TaskHarness {
    /// Enqueue this harness for its first (or next) poll.
    fn enqueue(self: Arc<Self>) {
        if self.single {
            let shared = Arc::clone(&self.shared);
            shared.push_async_ready(self);
        } else {
            let shared = Arc::clone(&self.shared);
            shared.push(Box::new(move || self.run_boxed()));
        }
    }

    /// Transition for a wake: queue the harness if idle, mark a re-poll if it is
    /// mid-poll, and otherwise do nothing. Returns whether to actually enqueue.
    fn note_wake(&self) -> bool {
        let mut inner = self.inner.lock().unwrap();
        match inner.run {
            RunState::Idle => {
                inner.run = RunState::Queued;
                true
            }
            RunState::Running => {
                inner.run = RunState::RunningNotified;
                false
            }
            RunState::Queued | RunState::RunningNotified | RunState::Done => false,
        }
    }
}

impl WakeTask for TaskHarness {
    fn wake_by_ref(self: &Arc<Self>) {
        if self.note_wake() {
            Arc::clone(self).enqueue();
        }
    }
}

impl RunnableTask for TaskHarness {
    fn run_boxed(self: Arc<Self>) {
        // Claim the future for polling (Queued -> Running).
        let mut future = {
            let mut inner = self.inner.lock().unwrap();
            if inner.run != RunState::Queued {
                return;
            }
            inner.run = RunState::Running;
            match inner.future.take() {
                Some(future) => future,
                None => {
                    inner.run = RunState::Done;
                    return;
                }
            }
        };

        let waker = waker_of(Arc::clone(&self));
        let mut cx = Context::from_waker(&waker);

        loop {
            match future.as_mut().poll(&mut cx) {
                Poll::Ready(()) => {
                    let mut inner = self.inner.lock().unwrap();
                    inner.run = RunState::Done;
                    // `future` drops here, releasing captured state.
                    return;
                }
                Poll::Pending => {
                    let mut inner = self.inner.lock().unwrap();
                    match inner.run {
                        RunState::Running => {
                            inner.future = Some(future);
                            inner.run = RunState::Idle;
                            return;
                        }
                        RunState::RunningNotified => {
                            // A wake arrived mid-poll: poll again right away
                            // rather than bouncing through the queue.
                            inner.run = RunState::Running;
                            drop(inner);
                        }
                        RunState::Idle | RunState::Queued | RunState::Done => {
                            // No other transition can reach here while we hold
                            // the sole `Running` claim; treat as finished.
                            inner.future = Some(future);
                            inner.run = RunState::Idle;
                            return;
                        }
                    }
                }
            }
        }
    }
}

/// Parks a thread for [`TaskPool::block_on`] and unparks it when woken.
struct ThreadNotify {
    signaled: Mutex<bool>,
    cvar: Condvar,
}

impl ThreadNotify {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            signaled: Mutex::new(false),
            cvar: Condvar::new(),
        })
    }

    /// Block until signaled, then clear the flag.
    fn wait(&self) {
        let mut signaled = self.signaled.lock().unwrap();
        while !*signaled {
            signaled = self.cvar.wait(signaled).unwrap();
        }
        *signaled = false;
    }

    /// Take and clear the current signal state without blocking.
    fn take_signal(&self) -> bool {
        let mut signaled = self.signaled.lock().unwrap();
        std::mem::replace(&mut signaled, false)
    }
}

impl WakeTask for ThreadNotify {
    fn wake_by_ref(self: &Arc<Self>) {
        let mut signaled = self.signaled.lock().unwrap();
        *signaled = true;
        self.cvar.notify_all();
    }
}

impl TaskPool {
    /// Spawn `future` onto the pool, returning a [`Task`] handle for its output.
    ///
    /// The future is polled as an ordinary job on the shared work-stealing pool
    /// (or, in the single-threaded fallback, from the ready queue that
    /// [`TaskPool::block_on`] drains). Await or [`TaskPool::block_on`] the
    /// returned handle for the value, or wait on [`Task::counter`] from a job.
    pub fn spawn_async<F>(&self, future: F) -> Task<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        let cell = ResultCell::<F::Output>::new();
        let cell_for_future = Arc::clone(&cell);
        let wrapped = async move {
            let output = future.await;
            cell_for_future.complete(output);
        };

        let harness = Arc::new(TaskHarness {
            inner: Mutex::new(HarnessInner {
                future: Some(Box::pin(wrapped)),
                run: RunState::Queued,
            }),
            shared: Arc::clone(self.shared_state()),
            single: self.is_single_threaded(),
        });
        harness.enqueue();

        Task::new(cell)
    }

    /// Drive `future` to completion on the calling thread and return its output.
    ///
    /// On a multi-threaded pool this parks the caller on a condvar while the
    /// pool's workers make progress, waking only when the future's waker fires.
    /// In the single-threaded fallback it drives the ready queue inline. The
    /// caller should not be a pool worker, so parking cannot starve the pool.
    pub fn block_on<F>(&self, future: F) -> F::Output
    where
        F: Future,
    {
        let notify = ThreadNotify::new();
        let waker = waker_of(Arc::clone(&notify));
        let mut cx = Context::from_waker(&waker);
        let mut future = Box::pin(future);

        if self.is_single_threaded() {
            self.block_on_single(future.as_mut(), &mut cx, &notify)
        } else {
            loop {
                if let Poll::Ready(output) = future.as_mut().poll(&mut cx) {
                    return output;
                }
                notify.wait();
            }
        }
    }

    /// Single-threaded `block_on`: interleave draining the ready queue with
    /// polling the target future. Fails fast (never hangs) if the future is
    /// pending with nothing left to run.
    fn block_on_single<F>(
        &self,
        mut future: Pin<&mut F>,
        cx: &mut Context<'_>,
        notify: &ThreadNotify,
    ) -> F::Output
    where
        F: Future,
    {
        let shared = self.shared_state();
        loop {
            while let Some(task) = shared.pop_async_ready() {
                task.run_boxed();
            }
            if let Poll::Ready(output) = future.as_mut().poll(cx) {
                return output;
            }
            // Nothing ran and the future did not wake itself: it is waiting on
            // something the single-threaded executor can never deliver.
            if shared.async_ready_is_empty() && !notify.take_signal() {
                panic!(
                    "block_on stalled: the future is pending with no runnable \
                     tasks (single-threaded executor cannot make progress)"
                );
            }
        }
    }
}
