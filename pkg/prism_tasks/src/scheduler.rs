//! Work-stealing scheduler shared state.
//!
//! M0 uses per-worker `Mutex<VecDeque>` deques plus a global injector queue.
//! This honors the Chase-Lev *shape* (owner pushes/pops the back, stealers take
//! the front) while remaining fully safe; the lock-free Chase-Lev upgrade is a
//! later performance milestone. Correctness goals for M0 are fork-join
//! completion with no deadlock, which this design guarantees via help-on-wait.
//!
//! ## M2 fibers (`fibers` feature)
//! When the `fibers` feature is on, a worker does not run a job directly;
//! instead it wraps the job in a [`FiberInner`](crate::fiber::FiberInner) and
//! switches onto it. If the job calls `wait` on a non-complete counter, the
//! fiber suspends back to the worker loop and is parked in the
//! [`WaitSet`](crate::fiber::wait_set::WaitSet); the worker keeps running other
//! jobs and resumes the fiber once its counter reaches zero. With the feature
//! off, `wait` keeps the help-on-wait busy loop unchanged.

use alloc::collections::VecDeque;
use alloc::sync::Arc;
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::Duration;

use crate::async_exec::RunnableTask;
use crate::job::Job;
use crate::priority::{PriorityCell, PriorityInbox};

#[cfg(feature = "fibers")]
use crate::fiber::stack::StackPool;
#[cfg(feature = "fibers")]
use crate::fiber::wait_set::WaitSet;

thread_local! {
    /// `(pool id, worker index)` for the current thread, if it is a worker of
    /// some pool. Used to route `spawn` to the local deque and to pick a steal
    /// origin while helping during `wait`.
    static WORKER: Cell<Option<(usize, usize)>> = const { Cell::new(None) };
}

pub(crate) struct Shared {
    /// One deque per worker; owner uses the back, stealers the front.
    deques: Vec<Mutex<VecDeque<Job>>>,
    /// Global MPMC FIFO queue for off-pool jobs and async harness dispatches.
    injector: Mutex<VecDeque<Job>>,
    /// Priority-ordered inbox for group jobs and explicitly prioritized futures;
    /// drained highest-priority-first ahead of the ordinary deques.
    priority_inbox: PriorityInbox,
    /// Ready-to-poll async task harnesses for the single-threaded fallback,
    /// drained by [`crate::TaskPool::block_on`] (multi-threaded pools re-enqueue
    /// harnesses as jobs instead, using the priority inbox when requested).
    async_ready: Mutex<AsyncReady>,
    async_cvar: Condvar,
    /// Count of jobs currently queued anywhere (for park/wake decisions).
    queued: AtomicUsize,
    /// Set during shutdown to drain workers.
    shutdown: AtomicBool,
    /// Park lock + condvar for idle workers.
    park: Mutex<()>,
    cvar: Condvar,
    /// Suspended fibers and the resume queue (fiber feature only).
    #[cfg(feature = "fibers")]
    wait_set: WaitSet,
    /// Reusable fiber stacks (fiber feature only).
    #[cfg(feature = "fibers")]
    stack_pool: StackPool,
    /// Optional worker->core pinning plan (M4). When present, each worker pins
    /// itself to its assigned core at startup (best-effort; see
    /// [`crate::affinity`]). `None` means run unpinned.
    affinity_plan: Option<crate::affinity::WorkerCorePlan>,
}

#[derive(Default)]
struct AsyncReady {
    /// Ordinary futures keep their existing FIFO policy.
    queue: VecDeque<Arc<dyn RunnableTask>>,
    /// Insertion order is retained so a live-priority scan breaks ties FIFO.
    prioritized: VecDeque<(Arc<PriorityCell>, Arc<dyn RunnableTask>)>,
    epoch: u64,
}

impl Shared {
    pub(crate) fn new(num_workers: usize) -> Self {
        Self::with_affinity(num_workers, None)
    }

    /// Build shared state, optionally carrying a worker->core pinning plan (M4).
    pub(crate) fn with_affinity(
        num_workers: usize,
        affinity_plan: Option<crate::affinity::WorkerCorePlan>,
    ) -> Self {
        let mut deques = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            deques.push(Mutex::new(VecDeque::new()));
        }
        let shared = Self {
            deques,
            injector: Mutex::new(VecDeque::new()),
            priority_inbox: PriorityInbox::new(),
            async_ready: Mutex::new(AsyncReady::default()),
            async_cvar: Condvar::new(),
            queued: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            park: Mutex::new(()),
            cvar: Condvar::new(),
            #[cfg(feature = "fibers")]
            wait_set: WaitSet::new(),
            #[cfg(feature = "fibers")]
            stack_pool: StackPool::new(),
            affinity_plan,
        };
        // M4: pre-warm the large-stack free list with one stack per worker so
        // the first fork-join frame reuses stacks instead of paying allocation
        // latency on its critical path (design §8/§12 预分配复用). Small stacks
        // stay cold until a shallow job asks for one.
        #[cfg(feature = "fibers")]
        shared
            .stack_pool
            .prewarm(crate::fiber::stack::StackClass::Large, num_workers);
        shared
    }

    pub(crate) fn num_workers(&self) -> usize {
        self.deques.len()
    }

    fn pool_id(&self) -> usize {
        core::ptr::from_ref::<Shared>(self) as usize
    }

    /// Enqueue a prioritized job or future dispatch under its
    /// shared priority `cell`, then wake one idle worker. Drained ahead of the
    /// ordinary deques by [`Self::find_task`], giving high-priority groups
    /// precedence and letting in-place boosts implement priority inheritance.
    pub(crate) fn push_prioritized(&self, cell: Arc<PriorityCell>, job: Job) {
        // Reserve the count *before* publishing so a worker cannot pop the job
        // and `fetch_sub` before the `fetch_add`, which underflowed `queued` to
        // `usize::MAX` and wedged shutdown (CR-found).
        self.queued.fetch_add(1, Ordering::Release);
        self.priority_inbox.push(cell, job);
        self.cvar.notify_one();
    }

    /// Push a job onto the local worker deque if the caller is a worker of this
    /// pool, otherwise onto the global injector. Wakes one idle worker.
    pub(crate) fn push(&self, job: Job) {
        // Reserve before publishing (see `push_prioritized`).
        self.queued.fetch_add(1, Ordering::Release);
        let local = WORKER.with(Cell::get);
        match local {
            Some((pid, idx)) if pid == self.pool_id() && idx < self.deques.len() => {
                self.deques[idx].lock().unwrap().push_back(job);
            }
            _ => {
                self.injector.lock().unwrap().push_back(job);
            }
        }
        self.cvar.notify_one();
    }

    /// Publish a job at the global FIFO tail even when called on a worker.
    /// Async poll continuations use this so a cooperative yield cannot turn
    /// into immediate local-LIFO re-entry ahead of ordinary injected jobs.
    /// Priority work and local deques retain precedence; FIFO applies only
    /// within the injector, not across all scheduler queues.
    pub(crate) fn push_injected(&self, job: Job) {
        // As with `push`, reserve before publishing to avoid count underflow.
        self.queued.fetch_add(1, Ordering::Release);
        self.injector.lock().unwrap().push_back(job);
        self.cvar.notify_one();
    }

    /// Obtain a job: live priority first, then local LIFO, injector FIFO, steal.
    pub(crate) fn find_task(&self, hint: Option<usize>) -> Option<Job> {
        // Prioritized dispatches run first (no preemption of a running poll).
        if let Some(job) = self.priority_inbox.pop_highest() {
            self.queued.fetch_sub(1, Ordering::Release);
            return Some(job);
        }
        if let Some(idx) = hint
            && let Some(dq) = self.deques.get(idx)
            && let Some(job) = dq.lock().unwrap().pop_back()
        {
            self.queued.fetch_sub(1, Ordering::Release);
            return Some(job);
        }
        if let Some(job) = self.injector.lock().unwrap().pop_front() {
            self.queued.fetch_sub(1, Ordering::Release);
            return Some(job);
        }
        for (j, dq) in self.deques.iter().enumerate() {
            if Some(j) == hint {
                continue;
            }
            if let Some(job) = dq.lock().unwrap().pop_front() {
                self.queued.fetch_sub(1, Ordering::Release);
                return Some(job);
            }
        }
        None
    }

    fn has_queued(&self) -> bool {
        self.queued.load(Ordering::Acquire) > 0
    }

    fn is_shutdown(&self) -> bool {
        self.shutdown.load(Ordering::Acquire)
    }

    /// The worker thread main loop. Dispatches to the fiber-aware or classic
    /// loop depending on the `fibers` feature.
    pub(crate) fn run_worker(&self, pool_id: usize, index: usize) {
        WORKER.with(|w| w.set(Some((pool_id, index))));
        // M4: pin this worker to its assigned core when a plan is present.
        // Best-effort — an unsupported platform (e.g. macOS) or an OS error
        // just leaves the worker unpinned; correctness never depends on it.
        if let Some(plan) = &self.affinity_plan {
            let _ = plan.pin_current(index);
        }
        #[cfg(feature = "fibers")]
        self.run_worker_fibers(index);
        #[cfg(not(feature = "fibers"))]
        self.run_worker_classic(index);
        WORKER.with(|w| w.set(None));
    }

    /// Classic (non-fiber) worker loop: run jobs directly, park when idle.
    #[cfg(not(feature = "fibers"))]
    fn run_worker_classic(&self, index: usize) {
        loop {
            if let Some(job) = self.find_task(Some(index)) {
                job();
                continue;
            }
            if self.is_shutdown() && !self.has_queued() {
                break;
            }
            // Nothing to do: park until woken or a short timeout elapses.
            let guard = self.park.lock().unwrap();
            if self.has_queued() || self.is_shutdown() {
                drop(guard);
                continue;
            }
            let _ = self
                .cvar
                .wait_timeout(guard, Duration::from_millis(10))
                .unwrap();
        }
    }

    /// Fiber-aware worker loop: resume ready fibers first, then run fresh jobs
    /// on fibers, flushing the wait-set after each unit of work.
    #[cfg(feature = "fibers")]
    fn run_worker_fibers(&self, index: usize) {
        loop {
            if let Some(fiber) = self.wait_set.pop_resume() {
                crate::fiber::run_fiber_switch(self, fiber.0);
                self.flush_waiters();
                continue;
            }
            if let Some(job) = self.find_task(Some(index)) {
                let fiber = crate::fiber::spawn_fiber(self, job);
                crate::fiber::run_fiber_switch(self, fiber);
                self.flush_waiters();
                continue;
            }
            // No immediately runnable work; try to unblock parked fibers.
            self.flush_waiters();
            if self.is_shutdown() && !self.has_queued() && self.wait_set.resume_is_empty() {
                break;
            }
            let guard = self.park.lock().unwrap();
            if self.has_queued() || !self.wait_set.resume_is_empty() || self.is_shutdown() {
                drop(guard);
                continue;
            }
            let _ = self
                .cvar
                .wait_timeout(guard, Duration::from_millis(5))
                .unwrap();
        }
    }

    /// Help run jobs until `done` returns true (classic, non-fiber build).
    /// Safe to call from a worker or an external thread; running jobs inline
    /// prevents fork-join deadlock when the dependency graph is deeper than the
    /// worker count.
    #[cfg(not(feature = "fibers"))]
    pub(crate) fn help_until(&self, mut done: impl FnMut() -> bool) {
        let hint = self.local_hint();
        while !done() {
            if let Some(job) = self.find_task(hint) {
                job();
            } else {
                std::thread::yield_now();
            }
        }
    }

    /// Help until `done` (fiber build). The calling thread is not itself on a
    /// fiber (a fiber's `wait` suspends instead of helping), so it drives the
    /// pool: resume ready fibers, run fresh jobs on fibers, and flush the
    /// wait-set, until `done` holds.
    #[cfg(feature = "fibers")]
    pub(crate) fn help_until(&self, mut done: impl FnMut() -> bool) {
        let hint = self.local_hint();
        while !done() {
            if let Some(fiber) = self.wait_set.pop_resume() {
                crate::fiber::run_fiber_switch(self, fiber.0);
                self.flush_waiters();
            } else if let Some(job) = self.find_task(hint) {
                let fiber = crate::fiber::spawn_fiber(self, job);
                crate::fiber::run_fiber_switch(self, fiber);
                self.flush_waiters();
            } else {
                self.flush_waiters();
                std::thread::yield_now();
            }
        }
    }

    /// The caller's worker index within this pool, if any (steal hint).
    fn local_hint(&self) -> Option<usize> {
        WORKER.with(Cell::get).and_then(|(pid, idx)| {
            if pid == self.pool_id() {
                Some(idx)
            } else {
                None
            }
        })
    }

    /// Move any now-complete parked fibers onto the resume queue and wake idle
    /// workers to pick them up.
    #[cfg(feature = "fibers")]
    fn flush_waiters(&self) {
        if self.wait_set.flush() {
            self.cvar.notify_all();
        }
    }

    /// Access the wait-set (fiber lifecycle helpers in [`crate::fiber`]).
    #[cfg(feature = "fibers")]
    pub(crate) fn wait_set(&self) -> &WaitSet {
        &self.wait_set
    }

    /// Access the fiber stack pool.
    #[cfg(feature = "fibers")]
    pub(crate) fn stack_pool(&self) -> &StackPool {
        &self.stack_pool
    }

    /// Wake one idle worker (best-effort; the parked-worker loop also polls on a
    /// short timeout, so a missed wake only costs latency, never correctness).
    #[cfg(feature = "fibers")]
    pub(crate) fn wake_one(&self) {
        self.cvar.notify_one();
    }

    /// Signal shutdown and wake all workers so they can drain and exit.
    pub(crate) fn begin_shutdown(&self) {
        self.shutdown.store(true, Ordering::Release);
        self.cvar.notify_all();
    }

    /// Enqueue a ready async task harness (single-threaded fallback only).
    pub(crate) fn push_async_ready(
        &self,
        task: Arc<dyn RunnableTask>,
        priority: Option<Arc<PriorityCell>>,
    ) {
        let mut ready = self.async_ready.lock().unwrap();
        if let Some(priority) = priority {
            ready.prioritized.push_back((priority, task));
        } else {
            ready.queue.push_back(task);
        }
        ready.epoch = ready.epoch.wrapping_add(1);
        self.async_cvar.notify_all();
    }

    /// Pop one ready async task harness, if any (single-threaded fallback).
    /// Explicit priority entries, even Background, precede the ordinary FIFO;
    /// this mirrors the worker inbox, not a unified Normal-priority queue.
    pub(crate) fn pop_async_ready(&self) -> Option<Arc<dyn RunnableTask>> {
        let mut ready = self.async_ready.lock().unwrap();
        if let Some((cell, _)) = ready.prioritized.front() {
            let mut best = 0;
            let mut priority = cell.get();
            for (index, (cell, _)) in ready.prioritized.iter().enumerate().skip(1) {
                let live = cell.get();
                if live > priority {
                    best = index;
                    priority = live;
                }
            }
            // Unlike swap-remove, removal retains FIFO order among ties after
            // earlier dispatches (and boosts). No task is run under the lock.
            return ready.prioritized.remove(best).map(|(_, task)| task);
        }
        ready.queue.pop_front()
    }

    /// Whether the single-threaded async ready queue is empty.
    pub(crate) fn async_ready_is_empty(&self) -> bool {
        let ready = self.async_ready.lock().unwrap();
        ready.queue.is_empty() && ready.prioritized.is_empty()
    }

    /// Observe before polling/draining so wake-before-wait cannot be lost.
    pub(crate) fn async_epoch(&self) -> u64 {
        self.async_ready.lock().unwrap().epoch
    }

    /// A directly-polled future may wake without adding a queued harness.
    pub(crate) fn signal_async_driver(&self) {
        let mut ready = self.async_ready.lock().unwrap();
        ready.epoch = ready.epoch.wrapping_add(1);
        self.async_cvar.notify_all();
    }

    /// Wait on the same mutex that publishes queue entries and direct wakes.
    /// Each driver retains its own epoch, so one waiter cannot consume another
    /// driver's notification. No user callbacks execute under this mutex.
    pub(crate) fn wait_async_ready(&self, epoch: u64, timeout: Duration) -> bool {
        let ready = self.async_ready.lock().unwrap();
        let (ready, _) = self
            .async_cvar
            .wait_timeout_while(ready, timeout, |ready| {
                ready.queue.is_empty() && ready.prioritized.is_empty() && ready.epoch == epoch
            })
            .unwrap();
        !ready.queue.is_empty() || !ready.prioritized.is_empty() || ready.epoch != epoch
    }

    pub(crate) fn id(&self) -> usize {
        self.pool_id()
    }

    /// The caller's worker index within this pool, if the current thread is one
    /// of its workers (M4 — lets the application index per-worker frame arenas).
    pub(crate) fn current_worker_index(&self) -> Option<usize> {
        self.local_hint()
    }

    /// The worker->core pinning plan, if this pool was built with pinning.
    pub(crate) fn affinity_plan(&self) -> Option<&crate::affinity::WorkerCorePlan> {
        self.affinity_plan.as_ref()
    }
}
