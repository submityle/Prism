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

use std::cell::Cell;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::Duration;

use crate::job::Job;

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
    /// Global MPMC injection queue for jobs spawned off-pool.
    injector: Mutex<VecDeque<Job>>,
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
}

impl Shared {
    pub(crate) fn new(num_workers: usize) -> Self {
        let mut deques = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            deques.push(Mutex::new(VecDeque::new()));
        }
        Self {
            deques,
            injector: Mutex::new(VecDeque::new()),
            queued: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            park: Mutex::new(()),
            cvar: Condvar::new(),
            #[cfg(feature = "fibers")]
            wait_set: WaitSet::new(),
            #[cfg(feature = "fibers")]
            stack_pool: StackPool::new(),
        }
    }

    pub(crate) fn num_workers(&self) -> usize {
        self.deques.len()
    }

    fn pool_id(&self) -> usize {
        self as *const Shared as usize
    }

    /// Push a job onto the local worker deque if the caller is a worker of this
    /// pool, otherwise onto the global injector. Wakes one idle worker.
    pub(crate) fn push(&self, job: Job) {
        let local = WORKER.with(|w| w.get());
        match local {
            Some((pid, idx)) if pid == self.pool_id() && idx < self.deques.len() => {
                self.deques[idx].lock().unwrap().push_back(job);
            }
            _ => {
                self.injector.lock().unwrap().push_back(job);
            }
        }
        self.queued.fetch_add(1, Ordering::Release);
        self.cvar.notify_one();
    }

    /// Try to obtain one job: local LIFO, then injector FIFO, then steal.
    pub(crate) fn find_task(&self, hint: Option<usize>) -> Option<Job> {
        if let Some(idx) = hint {
            if let Some(dq) = self.deques.get(idx) {
                if let Some(job) = dq.lock().unwrap().pop_back() {
                    self.queued.fetch_sub(1, Ordering::Release);
                    return Some(job);
                }
            }
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
            if self.has_queued()
                || !self.wait_set.resume_is_empty()
                || self.is_shutdown()
            {
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
        WORKER.with(|w| w.get()).and_then(|(pid, idx)| {
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

    pub(crate) fn id(&self) -> usize {
        self.pool_id()
    }
}
