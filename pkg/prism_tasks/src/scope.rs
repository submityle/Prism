//! Structured-parallelism scope (rayon-shaped `scope`).
//!
//! A [`Scope`] lets you spawn *borrowed* (non-`'static`) tasks and guarantees
//! that every spawned task has finished before [`TaskPool::scope`] returns. The
//! calling thread participates as a worker while waiting (see
//! [`TaskPool::wait`]), so nested scopes and deeper-than-worker-count fork-join
//! graphs cannot deadlock.
//!
//! ## Soundness
//! Spawning a borrowed closure requires erasing its lifetime so it can travel
//! through the `'static` job queue. This is sound because the scope *blocks
//! until its [`Counter`] drains* before returning — even if the scope body or a
//! child task panics — so no spawned task (and none of its borrows) can outlive
//! the scope. This is the same contract `std::thread::scope` upholds.

use std::any::Any;
use std::marker::PhantomData;
use std::panic::{self, AssertUnwindSafe};
use std::sync::Mutex;

use crate::job::Job;
use crate::{Counter, TaskPool};

/// A structured-parallelism scope bound to a [`TaskPool`].
///
/// Obtained via [`TaskPool::scope`]. Tasks spawned with [`Scope::spawn`] may
/// borrow from the enclosing stack frame (the `'env` environment); all of them
/// are joined before `scope` returns.
///
/// `'scope` is the lifetime of the scope itself and `'env` is the lifetime of
/// data borrowed from outside the scope.
pub struct Scope<'scope, 'env: 'scope> {
    pool: &'scope TaskPool,
    counter: Counter,
    /// First panic captured from a spawned task, re-raised after the join.
    panic: Mutex<Option<Box<dyn Any + Send + 'static>>>,
    /// Invariant over `'env`: borrowed data must strictly outlive the scope.
    _env: PhantomData<&'env mut &'env ()>,
    /// Invariant over `'scope`.
    _scope: PhantomData<&'scope mut &'scope ()>,
}

impl<'scope, 'env> Scope<'scope, 'env> {
    /// Spawn a task that may borrow from the enclosing `'env` environment.
    ///
    /// The task runs on the pool (or inline in the single-threaded fallback)
    /// and is guaranteed to complete before the owning [`TaskPool::scope`] call
    /// returns. Panics raised by the task are captured and re-raised on the
    /// scope-owning thread after all sibling tasks have joined.
    pub fn spawn<F>(&'scope self, f: F)
    where
        F: FnOnce() + Send + 'scope,
    {
        self.enqueue(&self.counter, f);
    }

    /// Fork-join two borrowed closures, returning both results.
    ///
    /// `b` is spawned onto the scope and `a` runs on the calling thread, which
    /// then helps drive the pool until `b` finishes. Unlike [`TaskPool::join`],
    /// the closures may borrow from the enclosing `'env` environment.
    pub fn join<A, B, RA, RB>(&'scope self, a: A, b: B) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send + 'scope,
        B: FnOnce() -> RB + Send + 'scope,
        RA: Send + 'scope,
        RB: Send + 'scope,
    {
        let rb_slot: Mutex<Option<RB>> = Mutex::new(None);
        let rb_ref = &rb_slot;
        let inner = Counter::new();
        self.enqueue(&inner, move || {
            *rb_ref.lock().unwrap() = Some(b());
        });
        let ra = a();
        self.pool.wait(&inner);
        let rb = rb_slot.lock().unwrap().take().expect("b did not complete");
        (ra, rb)
    }

    /// Shared enqueue path for [`Scope::spawn`] and [`Scope::join`].
    ///
    /// The task is tracked on `counter` and funnels any panic into the scope's
    /// panic slot. The closure lifetime `'a` may be shorter than `'scope` (as
    /// in `join`, where it borrows a local result slot); the caller must wait
    /// on `counter` before `'a` ends, which both callers do.
    fn enqueue<'a, F>(&'scope self, counter: &Counter, f: F)
    where
        F: FnOnce() + Send + 'a,
        'scope: 'a,
    {
        counter.add(1);
        let counter = counter.clone();
        let scope: &'scope Scope<'scope, 'env> = self;
        let job = move || {
            let result = panic::catch_unwind(AssertUnwindSafe(f));
            if let Err(payload) = result {
                let mut slot = scope.panic.lock().unwrap();
                if slot.is_none() {
                    *slot = Some(payload);
                }
            }
            counter.finish_one();
        };

        if self.pool.is_single_threaded() {
            job();
            return;
        }

        let boxed: Box<dyn FnOnce() + Send + 'a> = Box::new(job);
        #[expect(
            unsafe_code,
            reason = "scoped task lifetime erasure, joined before scope returns"
        )]
        let boxed: Job = {
            // SAFETY: We erase the `'a` lifetime so the job can travel through
            // the pool's `'static` job queue. The caller waits on `counter`
            // (helping run jobs) until this and every sibling job has run
            // `finish_one`, and `TaskPool::scope` additionally drains the scope
            // counter before returning, even on panic. Therefore the job — and
            // any borrows it captured — is fully executed and dropped before
            // those borrows end, so it never outlives them.
            unsafe { std::mem::transmute::<Box<dyn FnOnce() + Send + 'a>, Job>(boxed) }
        };
        self.pool.push_job(boxed);
    }
}

impl TaskPool {
    /// Open a structured-parallelism scope.
    ///
    /// Within `f`, use [`Scope::spawn`] to launch tasks that may borrow local
    /// data. Every spawned task is guaranteed to finish before this call
    /// returns. If the scope body or any spawned task panics, all siblings are
    /// still joined first and then one captured panic is re-raised.
    ///
    /// ```
    /// # use prism_tasks::TaskPool;
    /// let pool = TaskPool::with_threads(4);
    /// let mut data = vec![0u32; 8];
    /// pool.scope(|s| {
    ///     for (i, slot) in data.iter_mut().enumerate() {
    ///         s.spawn(move || *slot = i as u32 * 2);
    ///     }
    /// });
    /// assert_eq!(data, vec![0, 2, 4, 6, 8, 10, 12, 14]);
    /// ```
    pub fn scope<'env, F, T>(&self, f: F) -> T
    where
        F: for<'scope> FnOnce(&'scope Scope<'scope, 'env>) -> T,
    {
        let scope = Scope {
            pool: self,
            counter: Counter::new(),
            panic: Mutex::new(None),
            _env: PhantomData,
            _scope: PhantomData,
        };

        let result = panic::catch_unwind(AssertUnwindSafe(|| f(&scope)));

        // Join every spawned task before returning, even if `f` panicked. This
        // is what makes the lifetime erasure in `spawn` sound.
        self.wait(&scope.counter);

        match result {
            Ok(value) => {
                let captured = scope.panic.lock().unwrap().take();
                if let Some(payload) = captured {
                    panic::resume_unwind(payload);
                }
                value
            }
            Err(payload) => panic::resume_unwind(payload),
        }
    }
}
