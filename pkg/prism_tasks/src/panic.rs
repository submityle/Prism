//! Panic propagation across the fork-join boundary (design §17 panic 传播).
//!
//! A job that panics on a worker must not take down the worker thread or poison
//! the pool. [`TaskPool::spawn_catch`] wraps the job in
//! [`catch_unwind`](std::panic::catch_unwind), stores the
//! [`Result`](std::thread::Result), and hands back a [`JobHandle`]. Joining the
//! handle re-raises the panic *on the joining thread* (so the failure surfaces
//! at the structured join point, not silently swallowed), while every other
//! worker and the pool itself keep running normally.
//!
//! This mirrors the panic capture [`TaskPool::scope`](crate::TaskPool::scope)
//! already performs for scoped tasks, but exposes an explicit, pollable handle
//! with a typed result for callers that spawn a single fallible job and want
//! its value (or its panic) back.

use alloc::sync::Arc;
use std::panic::{self, AssertUnwindSafe};
use std::sync::Mutex;
use std::thread;

use crate::{Counter, TaskPool};

/// A handle to a panic-isolated job spawned by [`TaskPool::spawn_catch`].
///
/// Join it to retrieve the job's value, re-raising any panic on the joining
/// thread ([`JobHandle::join`]) or returning the raw
/// [`Result`](std::thread::Result) ([`JobHandle::join_catch`]).
pub struct JobHandle<T> {
    /// Pool used to drive work while joining.
    pool: TaskPool,
    /// Completion counter for this one job.
    counter: Counter,
    /// Where the job stores its captured result/panic.
    slot: Arc<Mutex<Option<thread::Result<T>>>>,
}

impl TaskPool {
    /// Spawn `f` with its panic caught, returning a [`JobHandle`] that yields
    /// `f`'s value (or re-raises its panic) when joined.
    ///
    /// Unlike [`TaskPool::spawn`], a panic in `f` never unwinds a worker: it is
    /// captured and surfaced only when the handle is joined. The pool stays
    /// fully usable afterward. In the single-threaded fallback `f` runs inline
    /// and its result is ready before this returns.
    ///
    /// ```
    /// # use prism_tasks::TaskPool;
    /// let pool = TaskPool::with_threads(4);
    /// let handle = pool.spawn_catch(|| 2 + 3);
    /// assert_eq!(handle.join(), 5);
    /// ```
    pub fn spawn_catch<F, T>(&self, f: F) -> JobHandle<T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        let counter = Counter::new();
        counter.add(1);
        let slot: Arc<Mutex<Option<thread::Result<T>>>> = Arc::new(Mutex::new(None));
        let slot_write = Arc::clone(&slot);
        let finish = counter.clone();
        let job = move || {
            let result = panic::catch_unwind(AssertUnwindSafe(f));
            *slot_write.lock().unwrap() = Some(result);
            finish.finish_one();
        };
        if self.is_single_threaded() {
            job();
        } else {
            self.push_job(Box::new(job));
        }
        JobHandle {
            pool: self.clone(),
            counter,
            slot,
        }
    }
}

impl<T> JobHandle<T> {
    /// Whether the job has finished (its result is ready without blocking).
    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.counter.is_complete()
    }

    /// Wait for the job and return its value, re-raising its panic on this
    /// thread if it panicked.
    ///
    /// # Panics
    /// Resumes unwinding with the job's original panic payload if `f` panicked.
    #[must_use]
    pub fn join(self) -> T {
        match self.join_catch() {
            Ok(value) => value,
            Err(payload) => panic::resume_unwind(payload),
        }
    }

    /// Wait for the job and return its raw [`Result`](std::thread::Result):
    /// `Ok` with the value, or `Err` with the captured panic payload.
    pub fn join_catch(self) -> thread::Result<T> {
        self.pool.wait(&self.counter);
        self.slot
            .lock()
            .unwrap()
            .take()
            .expect("job completed but left no result")
    }
}

impl<T> core::fmt::Debug for JobHandle<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("JobHandle")
            .field("finished", &self.is_finished())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use crate::TaskPool;

    #[test]
    fn join_returns_value() {
        let pool = TaskPool::with_threads(2);
        let handle = pool.spawn_catch(|| 21 * 2);
        assert_eq!(handle.join(), 42);
    }

    #[test]
    fn panic_is_captured_and_pool_survives() {
        let pool = TaskPool::with_threads(2);
        let handle = pool.spawn_catch(|| {
            panic!("boom");
        });
        let result = handle.join_catch();
        assert!(result.is_err());
        // The pool is still fully usable after a job panicked.
        let ok = pool.spawn_catch(|| 7);
        assert_eq!(ok.join(), 7);
    }

    #[test]
    fn join_re_raises_panic() {
        let pool = TaskPool::with_threads(2);
        let handle = pool.spawn_catch(|| {
            panic!("surfaced at join");
        });
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| handle.join()));
        assert!(caught.is_err());
    }

    #[test]
    fn many_panicking_jobs_do_not_poison_pool() {
        let pool = TaskPool::with_threads(4);
        let mut handles = Vec::new();
        for i in 0..32 {
            handles.push(pool.spawn_catch(move || {
                if i % 2 == 0 {
                    panic!("even panics");
                }
                i
            }));
        }
        let mut ok = 0;
        let mut err = 0;
        for h in handles {
            match h.join_catch() {
                Ok(_) => ok += 1,
                Err(_) => err += 1,
            }
        }
        assert_eq!(ok, 16);
        assert_eq!(err, 16);
        // Pool still works.
        assert_eq!(pool.spawn_catch(|| 99).join(), 99);
    }
}
