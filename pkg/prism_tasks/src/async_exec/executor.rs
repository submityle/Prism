//! The minimal future executor: it schedules futures as jobs on the existing
//! [`TaskPool`] and drives them with a hand-rolled waker.
//!
//! ## Model
//! Spawning a future wraps it in a non-generic [`TaskHarness`] (the typed
//! output is funneled into a [`ResultCell`] shared with the [`Task`] handle).
//! The harness is a [`WakeTask`]: when its waker fires it re-enqueues itself.
//! A multi-threaded pool runs a bounded poll quantum as a job on a worker (so a
//! future shares the work-stealing load balancer with the job graph, exactly as
//! the design intends); the single-threaded fallback keeps a ready queue that
//! [`TaskPool::block_on`] drives in bounded batches inline.
//!
//! ## Deadlock-freedom
//! A harness is enqueued at most once at a time (a small [`RunState`] machine
//! guards against double-enqueue and handles a wake that lands *during* a poll
//! by re-polling within the quantum or re-queuing at its end). Continuations
//! without an explicit priority use the global FIFO injector, not a worker's
//! local LIFO deque, so yielding cannot overtake already-injected jobs.
//! Priority work and local deques still
//! take precedence over the injector; this is not strict scheduler-wide FIFO.
//! This is cooperative: a single long-running poll cannot be preempted.
//! Explicitly prioritized futures use the existing priority inbox with a live
//! shared cell on every enqueue. These queues can starve lower-priority work
//! (including ordinary work); the poll quantum does not guarantee fairness
//! across priority levels.
//! [`TaskPool::block_on`] parks the calling thread on a condvar and is woken
//! by the future's own waker, so it
//! does not spin on an unwoken pending root. In the single-threaded fallback
//! it drives the ready queue and detects a stalled pending future after a
//! bounded wait when no work is runnable.

use alloc::sync::{Arc, Weak};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Condvar, Mutex};
use std::task::{Context, Poll};

use super::task::{ResultCell, Task};
use super::waker::{waker_of, WakeTask};
use crate::priority::PriorityCell;
use crate::scheduler::Shared;
use crate::TaskPool;

/// Maximum polls in one scheduler dispatch, even if every poll wakes itself.
const POLL_QUANTUM: usize = 32;

/// Maximum harness dispatches before a zero-worker driver polls its root again.
/// Together with `POLL_QUANTUM`, bounds each batch to 1024 harness polls, not
/// elapsed time. The directly-polled root is outside the harness quantum.
const SINGLE_READY_BATCH: usize = 32;

/// A type-erased, re-schedulable unit the scheduler can run without knowing the
/// future's output type.
pub(crate) trait RunnableTask: Send + Sync {
    /// Poll the underlying future for at most one cooperative quantum.
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
    /// Kept by identity across initial dispatch, external wakes and yields.
    /// `None` preserves the ordinary future's FIFO scheduling policy.
    priority: Option<Arc<PriorityCell>>,
    /// Completes the task's result cell with a captured panic payload if the
    /// future panics while being polled (so waiters never hang).
    on_panic: Box<dyn Fn(Box<dyn core::any::Any + Send>) + Send + Sync>,
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
            let priority = self.priority.clone();
            shared.push_async_ready(self, priority);
        } else if let Some(priority) = &self.priority {
            let shared = Arc::clone(&self.shared);
            let priority = Arc::clone(priority);
            shared.push_prioritized(priority, Box::new(move || self.run_boxed()));
        } else {
            let shared = Arc::clone(&self.shared);
            // A continuation put on the worker's local LIFO deque could
            // immediately win again forever, starving the injector. Use FIFO
            // for all harness enqueues, including wakes from a pool worker.
            shared.push_injected(Box::new(move || self.run_boxed()));
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

        for poll_index in 0..POLL_QUANTUM {
            // Isolate a panicking future: complete the result cell with the
            // panic so `block_on`/await re-raise it instead of hanging, and
            // never unwind the worker (CR-found).
            let poll = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                future.as_mut().poll(&mut cx)
            }));
            let poll = match poll {
                Ok(poll) => poll,
                Err(payload) => {
                    self.inner.lock().unwrap().run = RunState::Done;
                    (self.on_panic)(payload);
                    return;
                }
            };
            match poll {
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
                            if poll_index + 1 == POLL_QUANTUM {
                                // Publish the future and the Queued state
                                // together. Racing wakes then coalesce with
                                // this continuation; none can enqueue a
                                // second runner or lose the pending wake.
                                inner.future = Some(future);
                                inner.run = RunState::Queued;
                                drop(inner);
                                self.enqueue();
                                return;
                            }
                            // Coalesce wakes within the remaining quantum.
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
    single_driver: Option<Weak<Shared>>,
}

impl ThreadNotify {
    fn new(single_driver: Option<Weak<Shared>>) -> Arc<Self> {
        Arc::new(Self {
            signaled: Mutex::new(false),
            cvar: Condvar::new(),
            single_driver,
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
        drop(signaled);
        if let Some(shared) = self.single_driver.as_ref().and_then(Weak::upgrade) {
            shared.signal_async_driver();
        }
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
        self.spawn_future_inner(None, future)
    }

    /// Spawn an ordinary future, with the same FIFO policy as
    /// [`TaskPool::spawn_async`].
    pub fn spawn_future<F>(&self, future: F) -> Task<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        self.spawn_async(future)
    }

    /// Spawn a future sharing the live, monotonically boostable `priority`.
    ///
    /// Initial dispatch, wakeups and poll-quantum continuations all retain the
    /// same cell. Boosts affect queued dispatches without another wake, including
    /// on zero-worker pools driven by [`TaskPool::block_on`]. Equal-priority
    /// dispatches are FIFO in their queue; running polls are never preempted.
    /// Explicit priority and ordinary FIFO queues are separate; ordinary futures
    /// are not implicitly Normal. Prioritized dispatches precede ordinary work,
    /// even at Background, and can starve it or lower priorities if continually
    /// runnable.
    /// The quantum bounds poll count, not time or cross-priority starvation.
    ///
    /// A cell from [`crate::PriorityGroup::priority_cell`] shares inheritance
    /// with that group's jobs, but the future is tracked by its own
    /// [`Task::counter`], not by the group's completion counter.
    pub fn spawn_future_with_priority<F>(
        &self,
        priority: Arc<PriorityCell>,
        future: F,
    ) -> Task<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        self.spawn_future_inner(Some(priority), future)
    }

    fn spawn_future_inner<F>(
        &self,
        priority: Option<Arc<PriorityCell>>,
        future: F,
    ) -> Task<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        let cell = ResultCell::<F::Output>::new();
        let cell_for_future = Arc::clone(&cell);
        let cell_for_panic = Arc::clone(&cell);
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
            priority,
            on_panic: Box::new(move |payload| cell_for_panic.complete_panic(payload)),
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
        let single_driver = self
            .is_single_threaded()
            .then(|| Arc::downgrade(self.shared_state()));
        let notify = ThreadNotify::new(single_driver);
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

    /// Single-threaded `block_on`: interleave bounded batches of ready harnesses
    /// with polling the target future, even when the queue never becomes empty.
    /// Detect a pending future with no runnable work after a bounded wait.
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
            let epoch = shared.async_epoch();
            for _ in 0..SINGLE_READY_BATCH {
                let Some(task) = shared.pop_async_ready() else {
                    break;
                };
                task.run_boxed();
            }
            if let Poll::Ready(output) = future.as_mut().poll(cx) {
                return output;
            }
            if shared.async_ready_is_empty() && !notify.take_signal() {
                // Nothing runnable and no pending wake. Wait (bounded) for an
                // external waker — e.g. the `IoReactor` thread — to fire before
                // declaring a stall, so an externally-driven future can make
                // progress without the "never hangs" guarantee being lost: a
                // genuinely stuck future still panics after the timeout.
                if !shared.wait_async_ready(epoch, block_on_stall_timeout())
                    && shared.async_ready_is_empty()
                    && !notify.take_signal()
                {
                    panic!(
                        "block_on stalled: the future is pending with no runnable \
                         tasks (single-threaded executor cannot make progress)"
                    );
                }
            }
        }
    }
}

/// How long the single-threaded `block_on` waits for an external waker before
/// declaring the future stalled.
fn block_on_stall_timeout() -> core::time::Duration {
    core::time::Duration::from_secs(5)
}

#[cfg(test)]
#[path = "single_wake_tests.rs"]
mod single_wake_tests;

#[cfg(test)]
#[path = "fairness_tests.rs"]
mod fairness_tests;

#[cfg(test)]
#[path = "priority_tests.rs"]
mod priority_tests;

#[cfg(test)]
mod tests {
    use crate::TaskPool;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    #[test]
    fn panicking_future_does_not_hang_block_on() {
        // A future that panics must complete its result cell with the panic so
        // `block_on`/await re-raise it rather than hang (CR-found).
        let pool = TaskPool::with_threads(1);
        let task = pool.spawn_async(async { panic!("future boom") });
        let r = catch_unwind(AssertUnwindSafe(|| {
            pool.block_on(task);
        }));
        assert!(r.is_err());
    }
}
