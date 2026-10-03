//! Work-stealing scheduler shared state.
//!
//! M0 uses per-worker `Mutex<VecDeque>` deques plus a global injector queue.
//! This honors the Chase-Lev *shape* (owner pushes/pops the back, stealers take
//! the front) while remaining fully safe; the lock-free Chase-Lev upgrade is a
//! later performance milestone. Correctness goals for M0 are fork-join
//! completion with no deadlock, which this design guarantees via help-on-wait.

use std::cell::Cell;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};

use crate::job::Job;

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

    /// The worker thread main loop.
    pub(crate) fn run_worker(&self, pool_id: usize, index: usize) {
        WORKER.with(|w| w.set(Some((pool_id, index))));
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
                .wait_timeout(guard, std::time::Duration::from_millis(10))
                .unwrap();
        }
        WORKER.with(|w| w.set(None));
    }

    /// Help run jobs until `done` returns true. Safe to call from a worker or an
    /// external thread; running jobs inline prevents fork-join deadlock when the
    /// dependency graph is deeper than the worker count.
    pub(crate) fn help_until(&self, mut done: impl FnMut() -> bool) {
        let hint = WORKER.with(|w| w.get()).and_then(|(pid, idx)| {
            if pid == self.pool_id() {
                Some(idx)
            } else {
                None
            }
        });
        while !done() {
            if let Some(job) = self.find_task(hint) {
                job();
            } else {
                std::thread::yield_now();
            }
        }
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
